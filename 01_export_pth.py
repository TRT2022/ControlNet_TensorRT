'''
导出controlnet中各部分的torch模型
xj 2023-07-17
'''

import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

def get_state_dicts(ckpt_path, location='cpu'):
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

    model = instantiate_from_config(config.model).cpu()

    state_dicts = get_state_dicts('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda')

    # clip text encoder
    clip_config = config["model"]['params']['cond_stage_config']
    clip_model = instantiate_from_config(clip_config).cpu()

    clip_dicts =  {k: state_dicts["cond_stage_model."+k] for k in clip_model.state_dict()}
    clip_model.load_state_dict(clip_dicts)

    clip_model = clip_model

    torch.save(clip_model,"./models/clip_encoder.pth")
    print("save clip encoder success!!!")

    # vae
    vae_config = config["model"]['params']['first_stage_config']
    vae_model = instantiate_from_config(vae_config).cpu()

    vae_dicts =  {k:state_dicts["first_stage_model."+k] for k in vae_model.state_dict()}
    vae_model.load_state_dict(vae_dicts)
   
    torch.save(vae_model,"./models/vae_decoder.pth")
    print("save vae success!!!")

    # 暂时不转unet和controlnet
    # #controlNet
    # controlnet_config = config["model"]['params']['control_stage_config']
    # controlnet_model = instantiate_from_config(controlnet_config).cpu()

    # controlnet_dicts =  {k:state_dicts["control_model."+k] for k in controlnet_model.state_dict()}
    # controlnet_model.load_state_dict(controlnet_dicts)
    

    # torch.save(controlnet_model,"./models/controlnet.pth")
    # print("save controlnet success!!!")

    # # unet
    # unet_config = config["model"]['params']['unet_config']
    # unet_model = instantiate_from_config(unet_config).cpu()

    # unet_dicts =  {k:state_dicts["model.diffusion_model."+k] for k in unet_model.state_dict()}
    # unet_model.load_state_dict(unet_dicts)
    

    # torch.save(unet_model,"./models/unet.pth")
    # print("save unet success!!!")
    

def save_model():
    pass

if __name__ == "__main__":
    model = create_pt_model('./models/cldm_v15.yaml')
