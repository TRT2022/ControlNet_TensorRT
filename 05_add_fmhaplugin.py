
import onnx_graphsurgeon as gs
import numpy as np
import onnx
# import tensorrt as trt
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from collections import OrderedDict
from onnx import shape_inference
from cuda import cudart

inntrest_nodes={
    # Einsum id
    "0":{
        "node_q":"/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },
    "1":{
        "node_q":"/control_model/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/control_model/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/control_model/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/control_model/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },

    "7":{
        "node_q":"/unet/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/unet/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/unet/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/unet/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },
    "9":{
        "node_q":"/control_model/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/control_model/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/control_model/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/control_model/input_blocks.2/input_blocks.2.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },
    "15":{
        "node_q":"/unet/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/unet/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/unet/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/unet/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },

    "17":{
        "node_q":"/control_model/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/control_model/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/control_model/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/control_model/input_blocks.4/input_blocks.4.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },

    "23":{
        "node_q":"/unet/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/unet/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/unet/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/unet/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    },  

    "25":{
        "node_q":"/control_model/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/control_model/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/control_model/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/control_model/input_blocks.5/input_blocks.5.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    }, 

    "31":{
        "node_q":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_q/MatMul",
        "node_k":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_k/MatMul",
        "node_v":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_v/MatMul",
        "seq_len":1536,
        "matmul":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    }, 
    # "33":{
    #     "node_q":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_q/MatMul",
    #     "node_k":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_k/MatMul",
    #     "node_v":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_v/MatMul",
    #     "seq_len":1536,
    #     "matmul":"/unet/input_blocks.7/input_blocks.7.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul",
    # }, 


   

}

def insert_attention_plugin(graph, fused_qkv_idx, heads=8):
    # node_x  = [node for node in graph.nodes if node.name == "/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/norm1/Add_1"][0]

    node_q = [node for node in graph.nodes if node.name == "/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_q/MatMul"][0]
    node_k = [node for node in graph.nodes if node.name == "/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_k/MatMul"][0] 
    node_v = [node for node in graph.nodes if node.name == "/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_v/MatMul"][0]
    # Get weights of Q
    weights_q = node_q.inputs[1].values
    # Get weights of K
    weights_k = node_k.inputs[1].values
    # Get weights of V
    weights_v = node_v.inputs[1].values

    # Input number of channels to Q, K and V
    C = weights_k.shape[0]  # 320
    # Number of heads
    H = heads
    # Hidden dimension per head
    D = weights_k.shape[1] // H  # 40
    # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
    weights_qkv = np.dstack([weights_q.reshape(C, H, D), weights_k.reshape(C, H, D), weights_v.reshape(C, H, D)]).reshape(C, 3 * H * D)  # 320,3*8*40

    input_tensor = node_k.inputs[0]  # K and V have the same input
    # Q, K and V must have the same output which we feed into fmha plugin
    output_tensor_k = node_k.outputs[0]
    # Concat and interleave weights such that the output of fused QKV GEMM has [b, s, h, 3, d] shape
    constant_weights_qkv = gs.Constant("Weights_QKV_{}".format(fused_qkv_idx), np.ascontiguousarray(weights_qkv))

    # Created a fused node  output: 2x1536x320, 320x(3*8*40)
    fused_qkv_node = gs.Node(op="MatMul", name="MatMul_QKV_{}".format(fused_qkv_idx), inputs=[input_tensor, constant_weights_qkv], outputs=[output_tensor_k])
    graph.nodes.append(fused_qkv_node)

    # Connect the output of the fused node to the inputs of the nodes after Q, K and V
    node_q.o(0).inputs[0] = output_tensor_k
    node_k.o(0).inputs[0] = output_tensor_k
    node_v.o(0).inputs[0] = output_tensor_k

    node_q.outputs.clear()
    node_k.outputs.clear()
    node_v.outputs.clear()

    node_q.inputs.clear()
    node_k.inputs.clear()
    node_v.inputs.clear()
    graph.cleanup().toposort()


    output_qkv = fused_qkv_node.o().inputs[0]
    
    # Clear the inputs of the nodes that follow the QKV GEMM
    # to delete these subgraphs (it will be substituted by fMHA plugin)
    fused_qkv_node.outputs[0].outputs[2].inputs.clear()
    fused_qkv_node.outputs[0].outputs[1].inputs.clear()
    fused_qkv_node.outputs[0].outputs[0].inputs.clear()

    weights_qkv = fused_qkv_node.inputs[1].values
    dims_per_head = weights_qkv.shape[1] // (heads * 3)  #40  320*(3*8*40)

    # Reshape dims
    shape = gs.Constant("Shape_QKV_{}".format(fused_qkv_idx), np.ascontiguousarray(np.array([2, 1536, heads, 3, dims_per_head], dtype=np.int64)))

    # Reshape output tensor
    output_shape = gs.Variable("ReshapeQKV_{}".format(fused_qkv_idx), np.dtype(np.float16), None)

    # Create fMHA plugin
    reshape = gs.Node(op="Reshape", name="Reshape_{}".format(fused_qkv_idx), inputs=[output_qkv, shape], outputs=[output_shape])
    # Insert node
    graph.nodes.append(reshape)

    # Create fMHA plugin input: 2x1536x8x3x40
    output_final = gs.Variable("output_attention_{}".format(fused_qkv_idx), np.dtype(np.float16), None)  # 2*1536*8*40
    fmha = gs.Node(op="fMHA_V2", name="fMHA_{}".format(fused_qkv_idx), inputs=[output_shape], outputs=[output_final])
    # Insert node
    graph.nodes.append(fmha)

    node_output = [node for node in graph.nodes if node.name=="/unet/input_blocks.1/input_blocks.1.1/transformer_blocks.0/attn1/to_out/to_out.0/MatMul"][0]
    node_output.i().outputs.clear()
    
    shape_2 = gs.Constant("Shape_QKV_2_{}".format(fused_qkv_idx), np.ascontiguousarray(np.array([2, 1536, 320], dtype=np.int64)))
    reshape_2 = gs.Node(op="Reshape", name="Reshape_2_{}".format(fused_qkv_idx), inputs=[output_final, shape_2], outputs=[node_output.inputs[0]])
    graph.nodes.append(reshape_2)

    graph.cleanup().toposort()
    return graph


if __name__ == "__main__":

    graph = gs.import_onnx(onnx.load("./models/combine_0.onnx",load_external_data=True))

    num_heads = 8
    graph = insert_attention_plugin(graph, fused_qkv_idx=1, heads=8)
    

    # # fmhca
    # props = cudart.cudaGetDeviceProperties(0)[1]
    # sm = props.major * 10 + props.minor
    # num_fmhca_inserted = insert_fmhca_plugin(graph,num_heads, sm)
    # print('UNet: inserted '+str(num_fmhca_inserted)+' fMHCA plugins')


    onnx.save(gs.export_onnx(graph),"./models/combine_1.onnx",save_as_external_data=True)
