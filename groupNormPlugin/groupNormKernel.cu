/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "groupNormKernel.h"

#include <cmath>

#include <cub/cub.cuh>

static inline __device__ __host__ float sigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

struct GroupSums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sumSq;
};

struct GroupSumsOp
{
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b)
    {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

    // Allocate shared memory for BlockScan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.
    __shared__ float2 smem[tTHREADS_PER_BLOCK];

    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // The sums.
    float sum = 0.F;
    float sumSq = 0.F;

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The offset.
        int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Update the sum.
        sum += f2.x + f2.y;
        // Update the sum of squares.
        sumSq += f2.x * f2.x + f2.y * f2.y;
    }

    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

    // The data for the summations.
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

    // Do the segmented scan.
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

    // Store the results for the groups in shared memory (to produce coalesced
    // stores later).
    if (cj == params.cPerGroup - 2 /* 2 channels per thread */)
    {
        smem[gi] = make_float2(out.sum, out.sumSq);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The global group index.
    int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

    // Threads that have nothing left to do, exit.
    if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups)
    {
        return;
    }

    // The first threads (those storing to global memory, load the values).
    float2 sums = smem[threadIdx.x];

    // Store to global memory.
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNHWCSum(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the values are as we expect.
    // PLUGIN_ASSERT(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    // PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params); break;
    // default: PLUGIN_FAIL("Not implemented");
    }

    // PLUGIN_CUASSERT(cudaGetLastError());
}

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params)
{
    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = ci / params.cPerGroup;

    // Load the sum and sum of squares for the group.
    float sum = 0.F, sumSq = 0.F;
    if (gi < params.groups)
    {
        sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
        sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    }

    // Load gamma/beta.
    float2 gammaF2, betaF2;
    if (ci < params.c)
    {
        gammaF2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
        betaF2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);
    }

    // Compute the mean.
    float mean = sum * params.invHWC;
    // Compute the variance.
    float var = sumSq * params.invHWC - (mean * mean);
    // Compute the inverse of the stddev.
    float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The src/dst offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        if (ci < params.c)
        {
            h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
        }

        // Extract the two half values.
        float2 f2 = __half22float2(h2);

        // Normalize the channels.
        f2.x = (f2.x - mean) * invStdDev;
        f2.y = (f2.y - mean) * invStdDev;

        // Scale by gamma and add beta.
        f2.x = gammaF2.x * f2.x + betaF2.x;
        f2.y = gammaF2.y * f2.y + betaF2.y;

        // Apply Swish if needed.
        if (params.withSwish)
        {
            f2.x = f2.x * sigmoid(f2.x);
            f2.y = f2.y * sigmoid(f2.y);
        }

        // Store the scaled values.
        if (ci < params.c)
        {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
        }
    }
}

void groupNormNHWCScale(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the dimensions are aligned with what we expect.
    // PLUGIN_ASSERT(params.c % params.cPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    // PLUGIN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params); break;
    // default: PLUGIN_FAIL("Not implemented");
    }

    // PLUGIN_CUASSERT(cudaGetLastError());
}

//------------------



using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::GroupNormPlugin;
using nvinfer1::plugin::GroupNormPluginCreator;

namespace
{
static std::string const kGROUP_NORM_PLUGIN_NAME{"GroupNorm"};
static std::string const kGROUP_NORM_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float) + sizeof(int32_t)};
} // namespace

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor)
{
    int32_t maxDivisor = -1;
    for (int32_t i = 1; i <= std::sqrt(n); i++)
    {
        if (n % i == 0)
        {
            int32_t divisor1 = n / i;
            int32_t divisor2 = i;

            if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor)
            {
                maxDivisor = divisor1;
            }
            if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor)
            {
                maxDivisor = divisor2;
            }
        }
    }
    return maxDivisor;
}

// class GroupNormPlugin
GroupNormPlugin::GroupNormPlugin(std::string const& name, float epsilon, int32_t bSwish)
    : mName(name)
    , mEpsilon(epsilon)
    , mBSwish(bSwish)
{
    memset(&mParams, 0, sizeof(mParams));
}

GroupNormPlugin::GroupNormPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    // PLUGIN_VALIDATE(buffer != nullptr);
    // PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    auto const* d = static_cast<char const*>(buffer);
    auto const* a = d;

    mEpsilon = read<float>(d);
    mBSwish = read<int32_t>(d);

    // PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* GroupNormPlugin::clone() const noexcept
{
    try
    {
        auto p = new GroupNormPlugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return nullptr;
}

int32_t GroupNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType GroupNormPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    DataType ret{};
    try
    {
        // PLUGIN_VALIDATE(inputTypes != nullptr);
        // PLUGIN_VALIDATE(nbInputs > 0);
        ret = inputTypes[0];
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return ret;
}

DimsExprs GroupNormPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};
    try
    {
        // PLUGIN_VALIDATE(inputs != nullptr);
        // PLUGIN_VALIDATE(nbInputs > 0);
        ret = inputs[0];
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return ret;
}

bool GroupNormPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        // PLUGIN_VALIDATE(pos >= 0 && pos <= 3);
        if (pos == 0)
        {
            return inOut[0].type == DataType::kHALF && inOut[0].format == TensorFormat::kHWC8;
        }
        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        if (pos == 3)
        {
            return inOut[pos].format == inOut[0].format && inOut[pos].type == inOut[0].type;
        }
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return false;
}

void GroupNormPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t GroupNormPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return getWorkspaceSizeInBytes();
}

size_t GroupNormPlugin::getWorkspaceSizeInBytes() const
{
    return (sizeof(float) * 2) * 32 * 32; // sizeof(float2) * maxBatchSize * maxNumberOfGroup. float2
                                          // contians two buffers for sum and squared sum
}

int32_t GroupNormPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t cPerBlock = 320;
        int32_t maxBlocksPerHW = 1024;

        switch (inputDesc[0].dims.d[1])
        {
        case 960:
        case 1920: cPerBlock = 480; break;
        case 512:
        case 256: cPerBlock = 256; break;
        case 128: cPerBlock = 128; break;
        default: cPerBlock = 320;
        }

        mParams.withSwish = bool(mBSwish);
        mParams.dst = static_cast<half*>(outputs[0]);
        mParams.src = static_cast<half const*>(inputs[0]);
        mParams.gamma = static_cast<float const*>(inputs[1]);
        mParams.beta = static_cast<float const*>(inputs[2]);
        mParams.redBuffer = static_cast<float*>(workspace);
        mParams.n = inputDesc[0].dims.d[0];
        mParams.h = inputDesc[0].dims.d[2];
        mParams.w = inputDesc[0].dims.d[3];
        mParams.c = inputDesc[0].dims.d[1];
        mParams.groups = 32;
        mParams.hw = mParams.h * mParams.w;
        const int32_t blocksPerHW = findMaxDivisor(mParams.hw, maxBlocksPerHW);
        mParams.hwPerBlock = divUp(mParams.hw, blocksPerHW);
        mParams.cPerBlock = cPerBlock;
        mParams.cPerGroup = mParams.c / mParams.groups;
        mParams.hwc = mParams.hw * mParams.c;
        mParams.invHWC = 1.F / (float) (mParams.hw * mParams.cPerGroup);
        mParams.groupsPerBlock = cPerBlock / mParams.cPerGroup;

        cudaMemsetAsync(mParams.redBuffer, 0, getWorkspaceSizeInBytes(), stream);
        groupNormNHWCSum(mParams, stream);
        groupNormNHWCScale(mParams, stream);

        return 0;
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return -1;
}

void GroupNormPlugin::destroy() noexcept
{
    delete this;
}

int32_t GroupNormPlugin::initialize() noexcept
{
    return 0;
}

void GroupNormPlugin::terminate() noexcept {}

size_t GroupNormPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void GroupNormPlugin::serialize(void* buffer) const noexcept
{
    try
    {
        // PLUGIN_VALIDATE(buffer != nullptr);
        auto* d = static_cast<char*>(buffer);
        auto* a = d;
        write(d, mEpsilon); // float
        write(d, mBSwish);  // int32_t
        // PLUGIN_VALIDATE(d == a + getSerializationSize());
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
}

void GroupNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* GroupNormPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* GroupNormPlugin::getPluginType() const noexcept
{
    return kGROUP_NORM_PLUGIN_NAME.c_str();
}

char const* GroupNormPlugin::getPluginVersion() const noexcept
{
    return kGROUP_NORM_PLUGIN_VERSION.c_str();
}

// class GroupNormPluginCreator
PluginFieldCollection GroupNormPluginCreator::mFC{};
std::vector<PluginField> GroupNormPluginCreator::mPluginAttributes;

GroupNormPluginCreator::GroupNormPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bSwish", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

GroupNormPluginCreator::~GroupNormPluginCreator() {}

IPluginV2* GroupNormPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        float epsilon = 1e-5F;
        int32_t bSwish = 0;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            if (fc->fields[i].name == std::string("epsilon"))
            {
                epsilon = static_cast<float>(*(static_cast<float const*>((fc->fields[i].data))));
                continue;
            }
            if (fc->fields[i].name == std::string("bSwish"))
            {
                bSwish = static_cast<int32_t>(*(static_cast<int32_t const*>((fc->fields[i].data))));
                continue;
            }
        }
        return new GroupNormPlugin(name, epsilon, bSwish);
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return nullptr;
}

IPluginV2* GroupNormPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new GroupNormPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        // caughtError(e);
    }
    return nullptr;
}

char const* GroupNormPluginCreator::getPluginName() const noexcept
{
    return kGROUP_NORM_PLUGIN_NAME.c_str();
}

char const* GroupNormPluginCreator::getPluginVersion() const noexcept
{
    return kGROUP_NORM_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* GroupNormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

REGISTER_TENSORRT_PLUGIN(GroupNormPluginCreator);