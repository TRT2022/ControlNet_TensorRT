from collections import OrderedDict
import onnx
from cuda import cudart
from polygraphy.backend.onnx import modify_outputs, onnx_from_path, ModifyOutputs
import numpy as np

def clip_16(onnx_model,path):
    # change onnx -inf to -1e4
    for node in onnx_model.graph.node:
        # if node.name == "/text_model/ConstantOfShape_1":
        if node.op_type == "ConstantOfShape":
            # print(node)
            attr = node.attribute[0]
            # print(attr)
            if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
                np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
                print("raw array", np_array)
                np_array[np_array == -np.inf] = -1000000  # 将所有负无穷的值改为-1000000
                attr.t.raw_data = np_array.tobytes() 
                print("new array", np_array)
            # print(attr)
    onnx.save_model(onnx_model,path)



if __name__ == "__main__":

    onnx_model = onnx_from_path("./models/clip_encoder_0.onnx")
    clip_16(onnx_model,"./models/clip_encoder_1.onnx")