import onnx_graphsurgeon as gs
import numpy as np
import onnx
# import tensorrt as trt
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from collections import OrderedDict
from onnx import shape_inference

def insert_layernorm_plugin(graph):
    nLayerNormPlugin = 0
    for node in graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
                node.o().o(0).o().o().o().o().o().op == 'Mul' and \
                node.o().o(0).o().o().o().o().o().o().op == 'Add' and \
                len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1:

                # if node.i().op == "Add":
                #     inputTensor = node.inputs[0]  # CLIP
                # else:
                #     inputTensor = node.i().inputs[0]  # UNet and VAE

                inputTensor = node.inputs[0]

                # The first axis to normalize from can be inferred from the size of the `axes`
                # parameter of (any of) the `ReduceMean` node(s)
                reduceMeanNode = node.o().o(0).o()
                assert reduceMeanNode.op == "ReduceMean"
                firstNormAxis = -1 * np.size(np.array(reduceMeanNode.attrs["axes"]))

                gammaNode = node.o().o().o().o().o().o().o()
                index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
                gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float16)
                constantGamma = gs.Constant("LayerNormGamma-" + str(nLayerNormPlugin), np.ascontiguousarray(gamma.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                betaNode = gammaNode.o()
                index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
                beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float16)
                constantBeta = gs.Constant("LayerNormBeta-" + str(nLayerNormPlugin), np.ascontiguousarray(beta.reshape(-1)))

                inputList = [inputTensor, constantGamma, constantBeta]
                layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float16), inputTensor.shape)
                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, attrs=OrderedDict([('epsilon', 1.e-5), ('axis',firstNormAxis),("plugin_version","1")]), outputs=[layerNormV])
                graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1

                if betaNode.outputs[0] in graph.outputs:
                    index = graph.outputs.index(betaNode.outputs[0])
                    graph.outputs[index] = layerNormV
                else:
                    if betaNode.o().op == "Cast":
                        lastNode = betaNode.o()
                    else:
                        lastNode = betaNode
                    for subNode in graph.nodes:
                        if lastNode.outputs[0] in subNode.inputs:
                            index = subNode.inputs.index(lastNode.outputs[0])
                            subNode.inputs[index] = layerNormV
                    lastNode.outputs = []

    graph.cleanup().toposort()
    # # return nLayerNormPlugin
    # onnx.save(gs.export_onnx(graph),"./combine_0.onnx",save_as_external_data=True)
    print("layernorm")
    print(nLayerNormPlugin)
    return graph


def insert_layernorm_plugin_oneflow(graph):
    nLayerNormPlugin = 0
    for node in graph.nodes:
            if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
                node.o().o(0).o().o().o().o().o().op == 'Mul' and \
                node.o().o(0).o().o().o().o().o().o().op == 'Add' and \
                len(node.o().o(0).o().o().o().o().o().inputs[1].values.shape) == 1:

                # if node.i().op == "Add":
                #     inputTensor = node.inputs[0]  # CLIP
                # else:
                #     inputTensor = node.i().inputs[0]  # UNet and VAE

                inputTensor = node.inputs[0]

                # The first axis to normalize from can be inferred from the size of the `axes`
                # parameter of (any of) the `ReduceMean` node(s)
                reduceMeanNode = node.o().o(0).o()
                assert reduceMeanNode.op == "ReduceMean"
                firstNormAxis = -1 * np.size(np.array(reduceMeanNode.attrs["axes"]))

                gammaNode = node.o().o().o().o().o().o().o()
                index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
                gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float16)
                constantGamma = gs.Constant("LayerNormGamma-" + str(nLayerNormPlugin), np.ascontiguousarray(gamma.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

                betaNode = gammaNode.o()
                index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
                beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float16)
                constantBeta = gs.Constant("LayerNormBeta-" + str(nLayerNormPlugin), np.ascontiguousarray(beta.reshape(-1)))

                inputTensor.dtype = np.float16
                inputList = [inputTensor]
                # inputList = [inputTensor, constantGamma, constantBeta]


                layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float16), inputTensor.shape)
                # layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, attrs=OrderedDict([('epsilon', 1.e-5), ("plugin_version","5")]), outputs=[layerNormV])
                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, attrs=OrderedDict(), outputs=[layerNormV])
                graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1

                if betaNode.outputs[0] in graph.outputs:
                    index = graph.outputs.index(betaNode.outputs[0])
                    graph.outputs[index] = layerNormV
                else:
                    if betaNode.o().op == "Cast":
                        lastNode = betaNode.o()
                    else:
                        lastNode = betaNode
                    for subNode in graph.nodes:
                        if lastNode.outputs[0] in subNode.inputs:
                            index = subNode.inputs.index(lastNode.outputs[0])
                            subNode.inputs[index] = layerNormV
                    lastNode.outputs = []

    graph.cleanup().toposort()
    # # return nLayerNormPlugin
    # onnx.save(gs.export_onnx(graph),"./combine_0.onnx",save_as_external_data=True)
    print("layernorm")
    print(nLayerNormPlugin)
    return graph


# 明天试一下这个
def remove_casts(graph):
    nRemoveCastNode = 0
    for node in graph.nodes:
        # # Remove Cast nodes before qkv gemm
        # if node.op in ["Add", "Transpose"] and len(node.outputs[0].outputs) == 3 and node.o().op == "Cast" and node.o(1).op == "Cast" and node.o(2).op == "Cast":
        #     for i in range(len(node.outputs[0].outputs)):
        #         matMulNode = node.o(i, 0).o()
        #         matMulNode.inputs[0] = node.outputs[0]
        #         nRemoveCastNode += 1

        # # Remove double cast nodes after Softmax Node
        # if node.op == "Softmax" and node.o().op == "Cast" and node.o().o().op == "Cast":
        #     node.o().o().o().inputs[0] = node.outputs[0]
        #     nRemoveCastNode += 1
        
        if node.op == "Cast" and node.name != "/control_model/Cast":
            # print(graph.inputs)
            # print(node.inputs)
            
        
            for subNode in graph.nodes:
                if node.outputs[0] in subNode.inputs:
                    # if node.name == "/control_model/Cast_1":
                    index = subNode.inputs.index(node.outputs[0])
                    subNode.inputs[index] = node.inputs[0]
            node.outputs = []
            nRemoveCastNode += 1

        
    graph.cleanup().toposort()
    print("remove Cast")
    print(nRemoveCastNode)
    return graph
    # return nRemoveCastNode

# 明天试一下这个
def remove_parallel_swish(graph):
    mRemoveSwishNode = 0
    for node in graph.nodes:
        if node.op == "Gemm" and len(node.outputs[0].outputs) > 6:
            swishOutputTensor = None
            for nextNode in node.outputs[0].outputs:
                if nextNode.op == "Mul":
                    if swishOutputTensor is None:
                        swishOutputTensor = nextNode.outputs[0]
                    else:
                        nextGemmNode = nextNode.o(0)
                        assert nextGemmNode.op == "Gemm", "Unexpected node type for nextGemmNode {}".format(nextGemmNode.name)
                        nextGemmNode.inputs = [swishOutputTensor, nextGemmNode.inputs[1], nextGemmNode.inputs[2]]
                        nextNode.outputs.clear()
                        mRemoveSwishNode += 1

    graph.cleanup().toposort()
    print("remove Swish")
    print(mRemoveSwishNode)
    return graph

# 尝试完成有效
def decompose_instancenorms(graph):
    nRemoveInstanceNorm = 0
    for node in graph.nodes:
        if node.op == "InstanceNormalization":
            name = node.name + "/"
            input_tensor = node.inputs[0]
            output_tensor = node.outputs[0]
            mean_out = gs.Variable(name=name + "mean_out")
            mean_node = gs.Node(op="ReduceMean", name=name + "mean_node", attrs={"axes": [-1]}, inputs=[input_tensor], outputs=[mean_out])
            sub_out = gs.Variable(name=name + "sub_out")
            sub_node = gs.Node(op="Sub", name=name + "sub_node", attrs={}, inputs=[input_tensor, mean_out], outputs=[sub_out])
            pow_out = gs.Variable(name=name + "pow_out")
            pow_const = gs.Constant(name=name + "pow_const", values=np.array([2.0], dtype=np.float32))
            pow_node = gs.Node(op="Pow", name=name + "pow_node", attrs={}, inputs=[sub_out, pow_const], outputs=[pow_out])
            mean2_out = gs.Variable(name=name + "mean2_out")
            mean2_node = gs.Node(op="ReduceMean", name=name + "mean2_node", attrs={"axes": [-1]}, inputs=[pow_out], outputs=[mean2_out])
            epsilon_out = gs.Variable(name=name + "epsilon_out")
            epsilon_const = gs.Constant(name=name + "epsilon_const", values=np.array([node.attrs["epsilon"]], dtype=np.float32))
            epsilon_node = gs.Node(op="Add", name=name + "epsilon_node", attrs={}, inputs=[mean2_out, epsilon_const], outputs=[epsilon_out])
            sqrt_out = gs.Variable(name=name + "sqrt_out")
            sqrt_node = gs.Node(op="Sqrt", name=name + "sqrt_node", attrs={}, inputs=[epsilon_out], outputs=[sqrt_out])
            div_out = gs.Variable(name=name + "div_out")
            div_node = gs.Node(op="Div", name=name + "div_node", attrs={}, inputs=[sub_out, sqrt_out], outputs=[div_out])
            constantScale = gs.Constant("InstanceNormScaleV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(node.inputs[1].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
            constantBias = gs.Constant("InstanceBiasV-" + str(nRemoveInstanceNorm), np.ascontiguousarray(node.inputs[2].inputs[0].attrs["value"].values.reshape(1, 32, 1)))
            mul_out = gs.Variable(name=name + "mul_out")
            mul_node = gs.Node(op="Mul", name=name + "mul_node", attrs={}, inputs=[div_out, constantScale], outputs=[mul_out])
            add_node = gs.Node(op="Add", name=name + "add_node", attrs={}, inputs=[mul_out, constantBias], outputs=[output_tensor])
            graph.nodes.extend([mean_node, sub_node, pow_node, mean2_node, epsilon_node, sqrt_node, div_node, mul_node, add_node])
            node.inputs = []
            node.outputs = []
            nRemoveInstanceNorm += 1

    graph.cleanup().toposort()
    print("remove IN")
    print(nRemoveInstanceNorm)
    return graph

def insert_groupnorm_plugin(graph):
    nGroupNormPlugin = 0
    for node in graph.nodes:
        if node.op == "Reshape" and node.outputs != [] and \
            node.o().op == "ReduceMean" and node.o(1).op == "Sub" and node.o().o() == node.o(1) and \
            node.o().o().o().o().o().o().o().o().o().o().o().op == "Mul" and \
            node.o().o().o().o().o().o().o().o().o().o().o().o().op == "Add" and \
            len(node.o().o().o().o().o().o().o().o().inputs[1].values.shape) == 3 :
            # "node.outputs != []" is added for VAE
            # print(node)
            # print("--------------")
            assert len(node.outputs) == 1
            # inputTensor = node.outputs[0]
            inputTensor = node.inputs[0]
            # inputTensor = node.i().inputs[0]

            # print(inputTensor)

            gammaNode = node.o().o().o().o().o().o().o().o().o().o().o()
            # print(gammaNode)
            index = [type(i) == gs.ir.tensor.Constant for i in gammaNode.inputs].index(True)
            gamma = np.array(deepcopy(gammaNode.inputs[index].values.tolist()), dtype=np.float32)
            constantGamma = gs.Constant("groupNormGamma-" + str(nGroupNormPlugin), np.ascontiguousarray(gamma.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

            betaNode = gammaNode.o()
            index = [type(i) == gs.ir.tensor.Constant for i in betaNode.inputs].index(True)
            beta = np.array(deepcopy(betaNode.inputs[index].values.tolist()), dtype=np.float32)
            constantBeta = gs.Constant("groupNormBeta-" + str(nGroupNormPlugin), np.ascontiguousarray(beta.reshape(-1)))

            epsilon = node.o().o().o().o().o().inputs[1].values.tolist()[0]
            # print(epsilon)

            if betaNode.o().op == "Sigmoid":  # need Swish
                bSwish = True
                lastNode = betaNode.o().o()  # Mul node of Swish
            else:
                bSwish = False
                lastNode = betaNode  # Cast node after Group Norm

            if lastNode.o().op == "Cast":
                lastNode = lastNode.o()
            inputList = [inputTensor, constantGamma, constantBeta]
            groupNormV = gs.Variable("GroupNormV-" + str(nGroupNormPlugin), np.dtype(np.float16), inputTensor.shape)
            groupNormN = gs.Node("GroupNorm", "GroupNormN-" + str(nGroupNormPlugin), inputs=inputList, outputs=[groupNormV], attrs=OrderedDict([('epsilon', epsilon), ('bSwish', int(bSwish))]))
            graph.nodes.append(groupNormN)

            for subNode in graph.nodes:
                if lastNode.outputs[0] in subNode.inputs:
                    index = subNode.inputs.index(lastNode.outputs[0])
                    subNode.inputs[index] = groupNormV
            # node.i().inputs = []
            # print(lastNode)
            
            lastNode.outputs = []
            nGroupNormPlugin += 1

    graph.cleanup().toposort()
    # return nLayerNormPlugin
    # onnx.save(gs.export_onnx(graph),"./combine_0.onnx",save_as_external_data=True)
    print("GroupNorm")
    print(nGroupNormPlugin)
    return graph


# fix resize
def resize_fix(graph):
    '''
    This function loops through the graph looking for Resize nodes that uses scales for resize (has 3 inputs).
    It substitutes found Resize with Resize that takes the size of the output tensor instead of scales.
    It adds Shape->Slice->Concat
            Shape->Slice----^     subgraph to the graph to extract the shape of the output tensor.
    This fix is required for the dynamic shape support.
    '''
    mResizeNodes = 0
    for node in graph.nodes:
        if node.op == "Resize" and len(node.inputs) == 3:
            name = node.name + "/"
            
            add_node = node.o().o().i(1)
            div_node = node.i()
            
            shape_hw_out = gs.Variable(name=name + "shape_hw_out", dtype=np.int64, shape=[4])
            shape_hw = gs.Node(op="Shape", name=name+"shape_hw", inputs=[add_node.outputs[0]], outputs=[shape_hw_out])

            const_zero = gs.Constant(name=name + "const_zero", values=np.array([0], dtype=np.int64))
            const_two = gs.Constant(name=name + "const_two", values=np.array([2], dtype=np.int64))
            const_four = gs.Constant(name=name + "const_four", values=np.array([4], dtype=np.int64))

            slice_hw_out = gs.Variable(name=name + "slice_hw_out", dtype=np.int64, shape=[2])
            slice_hw = gs.Node(op="Slice", name=name+"slice_hw", inputs=[shape_hw_out, const_two, const_four, const_zero], outputs=[slice_hw_out])

            shape_bc_out = gs.Variable(name=name + "shape_bc_out", dtype=np.int64, shape=[2])
            shape_bc = gs.Node(op="Shape", name=name+"shape_bc", inputs=[div_node.outputs[0]], outputs=[shape_bc_out])

            slice_bc_out = gs.Variable(name=name + "slice_bc_out", dtype=np.int64, shape=[2])
            slice_bc = gs.Node(op="Slice", name=name+"slice_bc", inputs=[shape_bc_out, const_zero, const_two, const_zero], outputs=[slice_bc_out])

            concat_bchw_out = gs.Variable(name=name + "concat_bchw_out", dtype=np.int64, shape=[4])
            concat_bchw = gs.Node(op="Concat", name=name+"concat_bchw", attrs={"axis": 0}, inputs=[slice_bc_out, slice_hw_out], outputs=[concat_bchw_out])

            none_var = gs.Variable.empty()

            resize_bchw = gs.Node(op="Resize", name=name+"resize_bchw", attrs=node.attrs, inputs=[node.inputs[0], none_var, none_var, concat_bchw_out], outputs=[node.outputs[0]])

            graph.nodes.extend([shape_hw, slice_hw, shape_bc, slice_bc, concat_bchw, resize_bchw])

            node.inputs = []
            node.outputs = []

            mResizeNodes += 1

    graph.cleanup().toposort()

    print("mResizeNodes")
    print(mResizeNodes)
    return graph

def adjustAddNode(graph):
    nAdjustAddNode = 0
    for node in graph.nodes:
        # Change the bias const to the second input to allow Gemm+BiasAdd fusion in TRT.
        if node.op in ["Add"] and isinstance(node.inputs[0], gs.ir.tensor.Constant):
            tensor = node.inputs[1]
            bias = node.inputs[0]
            node.inputs = [tensor, bias]
            nAdjustAddNode += 1

    graph.cleanup().toposort()

    print("nAdjustAddNode")
    print(nAdjustAddNode)
    return graph


def infer_shapes(graph,return_onnx=False):
    onnx_graph = gs.export_onnx(graph)
    # if onnx_graph.ByteSize() > 2147483648:
    #     raise TypeError("ERROR: model size exceeds supported 2GB limit")
    # else:
    onnx_graph = shape_inference.infer_shapes(onnx_graph)

    graph = gs.import_onnx(onnx_graph)
    return graph

def insert_splitgelu_plugin(graph):
    nSplitGeLUPlugin = 0
    for node in graph.nodes:
        if node.op == "Erf":
            inputTensor = node.i().i().i().outputs[0]
            lastNode = node.o().o().o().o()
            # outputShape = inputTensor.shape
            # outputShape[2] = outputShape[2] // 2

            # splitGeLUV = gs.Variable("splitGeLUV-" + str(nSplitGeLUPlugin), np.dtype(np.float32), outputShape)
            splitGeLUV = gs.Variable("splitGeLUV-" + str(nSplitGeLUPlugin), np.dtype(np.float16))

            splitGeLUN = gs.Node("SplitGeLU", "splitGeLUN-" + str(nSplitGeLUPlugin), inputs=[inputTensor], outputs=[splitGeLUV])
            graph.nodes.append(splitGeLUN)

            for subNode in graph.nodes:
                if lastNode.outputs[0] in subNode.inputs:
                    index = subNode.inputs.index(lastNode.outputs[0])
                    subNode.inputs[index] = splitGeLUV
            lastNode.outputs = []
            nSplitGeLUPlugin += 1

    graph.cleanup().toposort()

    print("nSplitGeLUPlugin")
    print(nSplitGeLUPlugin)
    return graph


def insert_seq2spatial_plugin(graph):
    nSeqLen2SpatialPlugin = 0
    for node in graph.nodes:
        # if node.o().op == "Conv" and node.i().op=="Reshape": #and node.i().i(0).op=="Add" and node.i().i(0).i(0).op=="Add":
        if node.op == "Transpose" and node.o().o().op == "Conv" and node.o().op=="Reshape":
 
            transposeNode = node
            reshapeNode = node.i()
            assert reshapeNode.op == "Reshape", "Unexpected node type for reshapeNode {}".format(reshapeNode.name)
            residualNode = reshapeNode.i(0)
            assert residualNode.op == "Add", "Unexpected node type for residualNode {}".format(residualNode.name)
            biasNode = residualNode.i(0)
            assert biasNode.op == "Add", "Unexpected node type for biasNode {}".format(biasNode.name)
            biasIndex = [type(i) == gs.ir.tensor.Constant for i in biasNode.inputs].index(True)
            bias = np.array(deepcopy(biasNode.inputs[biasIndex].values.tolist()), dtype=np.float16)
            biasInput = gs.Constant("AddAddSeqLen2SpatialBias-" + str(nSeqLen2SpatialPlugin), np.ascontiguousarray(bias.reshape(-1)))
            inputIndex = 1 - biasIndex
            inputTensor = biasNode.inputs[inputIndex]
            residualInput = residualNode.inputs[1]
            outputTensor = transposeNode.outputs[0]
            outputShapeTensor = transposeNode.i().i().i(1).i(1).i(1).i().inputs[0]
            seqLen2SpatialNode = gs.Node("SeqLen2Spatial", "AddAddSeqLen2Spatial-" + str(nSeqLen2SpatialPlugin),
                inputs=[inputTensor, biasInput, residualInput, outputShapeTensor], outputs=[outputTensor])
            graph.nodes.append(seqLen2SpatialNode)
            # biasNode.inputs.clear()
            transposeNode.outputs.clear()
            nSeqLen2SpatialPlugin += 1
            # print(node)

            

    graph.cleanup().toposort()
    print("nSeqLen2SpatialPlugin")
    print(nSeqLen2SpatialPlugin)
    return graph




if __name__ == "__main__":
    graph = gs.import_onnx(onnx.load("./models/combine.onnx",load_external_data=True))

    # graph = insert_layernorm_plugin(graph) # 不work
    # graph = insert_layernorm_plugin_oneflow(graph) # 不work

    graph = remove_casts(graph)
    graph = remove_parallel_swish(graph)
    graph = decompose_instancenorms(graph)
    graph = insert_groupnorm_plugin(graph)
    # fix size
    graph = resize_fix(graph)
    graph = adjustAddNode(graph)

    # # splitgelu
    # # graph = infer_shapes(graph)
    # graph = insert_splitgelu_plugin(graph)  #不work

    # # insert_seq2spatial_plugin
    # graph = insert_seq2spatial_plugin(graph) #不work

    #TODO: FMHA,FMCA
    onnx.save(gs.export_onnx(graph),"./models/combine_0.onnx",save_as_external_data=True)


    # vae
    print("vae")
    graph = gs.import_onnx(onnx.load("./models/vae_decoder.onnx"))
    graph = infer_shapes(graph)
    graph = remove_casts(graph)
    # graph = resize_fix(graph)
    graph = adjustAddNode(graph)
    graph = remove_parallel_swish(graph)
    graph = decompose_instancenorms(graph)
    graph = insert_groupnorm_plugin(graph)
    onnx.save(gs.export_onnx(graph),"./models/vae_decoder_0.onnx")

    print("clip")
    graph = gs.import_onnx(onnx.load("./models/clip_encoder.onnx"))
    # graph = insert_layernorm_plugin(graph) # 不work
    graph = remove_casts(graph)
    graph = resize_fix(graph)
    graph = adjustAddNode(graph)
    graph = remove_parallel_swish(graph)
    graph = decompose_instancenorms(graph)
    graph = insert_groupnorm_plugin(graph)
    onnx.save(gs.export_onnx(graph),"./models/clip_encoder_0.onnx")
