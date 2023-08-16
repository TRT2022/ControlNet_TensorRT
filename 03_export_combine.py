'''
导出controlnet中各部分的torch模型
xj 2023-07-17
'''

import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

import torchvision
import onnx # 用于验证onnx模型
# import onnxruntime as ort  # 也可以使用onnxruntime 来做推断

import time
device=torch.device("cuda")

# class CombineUnetControlModel(nn.Module):
#     def __init__(self, unet, control_model):
#         super().__init__()
#         self.unet = unet
#         self.control_model = control_model

#     # (self, x, hint, timesteps, context, **kwargs):
#     # self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs
#     # ["x_noisy","hint","timesteps","context"] + [f"control_{i}" for i in range(13)]
#     def forward(self, x, hint, timesteps, context, control, **kwargs):
#         unet_out = self.unet(x, timesteps=timesteps, context=context, control=control,
#                              only_mid_control=False, **kwargs)
#         control_out = self.control_model(x, hint, timesteps, context, **kwargs)

#         return unet_out, control_out

class CombineUnetControlModel2(nn.Module):
    def __init__(self, unet, control_model):
        super().__init__()
        self.unet = unet
        self.control_model = control_model

    # (self, x, hint, timesteps, context, **kwargs):
    # self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs
    # ["x_noisy","hint","timesteps","context"] + [f"control_{i}" for i in range(13)]
    def forward(self, x, hint, timesteps, context, **kwargs):
        control = self.control_model(x, hint, timesteps, context, **kwargs)
        unet_out = self.unet(x, timesteps=timesteps, context=context, control=control,
                             only_mid_control=False, **kwargs)
        return unet_out


def get_state_dict(d):
    return d.get('state_dict', d)

def get_state_dicts(ckpt_path, location='cuda'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_pt_model(config_path):
    config = OmegaConf.load(config_path)

    model = instantiate_from_config(config.model).cuda()
    # model.half()

    state_dicts = get_state_dicts('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda')

    # CombineModel = CombineUnetControlModel(unet_model, controlnet_model)
    # torch.save(CombineModel,"./models/combine.pth")
    # print("save CombineNet success!!!")

    #controlNet
    controlnet_config = config["model"]['params']['control_stage_config']
    controlnet_model = instantiate_from_config(controlnet_config)

    controlnet_dicts =  {k:state_dicts["control_model."+k] for k in controlnet_model.state_dict()}
    controlnet_model.load_state_dict(controlnet_dicts)
    controlnet_model = controlnet_model.to(device)
    
    # unet
    unet_config = config["model"]['params']['unet_config']
    unet_model = instantiate_from_config(unet_config)

    unet_dicts =  {k:state_dicts["model.diffusion_model."+k] for k in unet_model.state_dict()}
    unet_model.load_state_dict(unet_dicts)
    unet_model = unet_model.to(device)
    
    with torch.no_grad():
    
        CombineModel = CombineUnetControlModel2(unet_model, controlnet_model)
        torch.save(CombineModel,"./models/combine.pth")
        print("save CombineNet success!!!")

        combine_model = CombineModel
        combine_model.to(device)
        combine_model.eval()

        input_names = ["x_noisy", "hint", "timesteps", "context"]
        output_names = ["unet_out"]

        dummpy_inputs = (torch.randn(2,4,32,48).cuda(),torch.randn(2,3, 256, 384).cuda(), torch.tensor([2]).cuda(),torch.randn(2, 77, 768).cuda())
        
        torch.onnx.export(combine_model, dummpy_inputs, "./models/combine.onnx", verbose=True,input_names=input_names, output_names=output_names,
            opset_version=13)
        
    #     print("combine_model torch2onnx success!!!")
        

def save_model():
    pass

if __name__ == "__main__":
    model = create_pt_model('./models/cldm_v15.yaml')
