from collections import OrderedDict
from copy import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import math
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
import random
from scipy import integrate
import tensorrt as trt
import torch
import requests
from io import BytesIO
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
groupnorm = ctypes.CDLL("./groupNormPlugin/groupNormKernel.so")


# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

def build(onnx_path,engine_path, fp16=True, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, workspace_size=3000000000):
        print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs['preview_features'] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs['preview_features'].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=engine_path)

if __name__ == "__main__":
    build("./models/combine_0.onnx","./combine.plan", fp16=True)
