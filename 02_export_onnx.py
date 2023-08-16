
import torch
import torchvision
import onnx # 用于验证onnx模型
# import onnxruntime as ort  # 也可以使用onnxruntime 来做推断

import time
device=torch.device("cpu")

# clip encoder
def torxh2onnx_clip(pth_path="./models/clip_encoder.pth",onnx_path="./models/clip_encoder.onnx"):
    # load torch pt model
    model = torch.load(pth_path)
    model.to(device)
    model.eval()

    print(model)

    input_names =["input_ids"]
    output_names = ['clip_encoder']


    dummy_input = torch.tensor([1]*77).reshape(1,77)
    # dummy_input = torch.randn(1, 3, 256, 256)

    # torch.onnx.export(model,dummy_input, onnx_path, verbose=True)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True,input_names=input_names,output_names=output_names,
                dynamic_axes= {
                            input_names[0]: {0: 'batch_size'},
                            output_names[0]: {0: 'batch_size'}},
                opset_version=18
                    )
    print("clip torch2onnx success!!!")
    
    

# vae decoder
def torch2onnx_vae(pth_path="./models/vae_decoder.pth",onnx_path="./models/vae_decoder.onnx"):
    # load torch pt model
    model = torch.load(pth_path)
    model.to(device)
    model.eval()

    print(model)

    # ----------->>dec
    # torch.Size([1, 4, 32, 48])
    # torch.Size([1, 3, 256, 384])
    # MMMMMMMMM


    input_names =["image"]
    output_names = ['dec']

    dummy_input = torch.randn(1, 4, 32, 48)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True,input_names=input_names,
        output_names=output_names,opset_version=17)
    print("vae torch2onnx success!!!")

    
# 暂时不做controlnet 和 UNet
# #controlnet
# def torch2onnx_controlnet(pth_path="./models/controlnet.pth",onnx_path="./models/controlnet.onnx"):
#     # load torch pt model

#     with torch.no_grad():
#         model = torch.load(pth_path)
#         model.to(device)
#         model.eval()

#         # print(model)
#         # print(x_noisy.size())
#         # print(torch.cat(cond['c_concat'], 1).size())
#         # print(t.size())
#         # print(cond_txt.size())
#         # print(control[0].size())
#         # print(len(control))
#         # print("-------end")

#         # ------------control net
#         # torch.Size([1, 4, 32, 48])
#         # torch.Size([1, 3, 256, 384])  
#         # torch.Size([1])
#         # torch.Size([1, 77, 768])
#         # torch.Size([1, 320, 32, 48])
#         # 13


#         input_names =["x_noisy","hint","timesteps","context"]
#         output_names = [f"control_{i}" for i in range(13)]

#         dummpy_inputs = (torch.randn(2,4,32,48),torch.randn(2,3, 256, 384),torch.tensor([2]),torch.randn(2, 77, 768))

#         torch.onnx.export(model, dummpy_inputs, onnx_path, verbose=True,input_names=input_names,output_names=output_names,#keep_initializers_as_inputs=True,
#             opset_version=17)
#         print("controlnet torch2onnx success!!!")
        

# #unet
# def torch2onnx_unet(pth_path="./models/unet.pth",onnx_path="./models/unet.onnx"):
#     # load torch pt model

#     with torch.no_grad():
#         model = torch.load(pth_path)
#         model.to(device)
#         model.eval()


#         # print("========unet")
#         # print(x.size())          x_noise       
#         # print(timesteps.size())  contronet timestep
#         # print(context.size())    controlnet context
#         # print(len(control))      controlnet生成的control
#         # print(control[0].size())
#         # print(only_mid_control)        False
#         # print(self.out(h).size())
#         # print("-----------------------------")

#         # ========unet
#         # torch.Size([1, 4, 32, 48])
#         # torch.Size([1])
#         # torch.Size([1, 77, 768])
#         # 13
#         # torch.Size([1, 320, 32, 48])
#         # False
#         # torch.Size([1, 4, 32, 48])
#         # -----------------------------


#         input_names =["x_noisy","timesteps","context"]+[f"control_{i}" for i in range(13)]
#         output_names = ["unet_out"] # torch.Size([1, 4, 32, 48])

#         dummpy_inputs = (torch.randn(2,4,32,48),torch.tensor([2]),torch.randn(2, 77, 768),
#             [
#                 torch.randn([2, 320, 32, 48]),
#                 torch.randn([2, 320, 32, 48]),
#                 torch.randn([2, 320, 32, 48]),
#                 torch.randn([2, 320, 16, 24]),
#                 torch.randn([2, 640, 16, 24]),
#                 torch.randn([2, 640, 16, 24]),
#                 torch.randn([2, 640, 8, 12]),
#                 torch.randn([2, 1280, 8, 12]),
#                 torch.randn([2, 1280, 8, 12]),
#                 torch.randn([2, 1280, 4, 6]),
#                 torch.randn([2, 1280, 4, 6]),
#                 torch.randn([2, 1280, 4, 6]),
#                 torch.randn([2, 1280, 4, 6]),
#             ]
            
#             )

#         torch.onnx.export(model, dummpy_inputs, onnx_path, verbose=True,input_names=input_names,output_names=output_names,
#             opset_version=17)
        
#         print("unet torch2onnx success!!!")



if __name__ == "__main__":
    torxh2onnx_clip()
    torch2onnx_vae()
    # torch2onnx_controlnet()
    # torch2onnx_unet()

