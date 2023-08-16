from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from ddim_hacked_trt import DDIMSampler
from ddim_hacked_trt import *

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

from trt_util import *
import time


class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        # self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()

        self.device = torch.device("cuda")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # # # self.controlnet_context = get_engine("./controlnet.plan")
        # # # self.unet_context = get_engine("./unet.plan")
        # self.combine_context = get_engine("./combine.plan")
        # self.postnet_context = get_engine("./postnet.plan")
        # self.ddim_sampler = DDIMSampler(self.model,controlnet_context=self.controlnet_context,unet_context=self.unet_context,postnet_context=self.postnet_context)
        
        # clip_encoder
        self.clip_engine = Engine("./clip_encoder.plan")
        self.clip_engine.load()
        self.clip_engine.activate(reuse_device_memory=None)
        self.clip_engine.allocate_buffers()
        # self.clip_context = get_engine("./clip_encoder.plan",dynamic=True)

        # combine
        self.combine_engine = Engine("./combine.plan")
        self.combine_engine.load()
        self.combine_engine.activate(reuse_device_memory=None)
        self.combine_engine.allocate_buffers()

        #vae
        self.vae_engine = Engine("./vae_decoder.plan")
        self.vae_engine.load()
        self.vae_engine.activate(reuse_device_memory=None)
        self.vae_engine.allocate_buffers()
        # self.vae_context = get_engine("./vae_decoder.plan")

        # #post
        # self.post_engine = Engine("./postnet.plan")
        # self.post_engine.load()
        # self.post_engine.activate(reuse_device_memory=None)
        # self.post_engine.allocate_buffers()
        self.post_engine = get_engine("./postnet.plan")

        
        self.ddim_sampler = DDIMSampler(self.model,combine_context=self.combine_engine,postnet_context=self.post_engine)

        self.model.control_scales = [1]*13

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        ddim_steps = 10  # 为了测试想法
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=False)

            #  # 调用clip encoder
            # cond_text = [prompt + ', ' + a_prompt]
            # cond_batch_encoding = self.tokenizer(cond_text, truncation=True, max_length=77, return_length=True,
            #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            # cond_tokens = cond_batch_encoding["input_ids"].to(self.device)

            # uncond_text =[n_prompt]
            # uncond_batch_encoding = self.tokenizer(uncond_text, truncation=True, max_length=77, return_length=True,
            #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            # uncond_tokens = uncond_batch_encoding["input_ids"].to(self.device)

            # buffer_cilp_D = [cond_tokens.data_ptr(),self.ddim_sampler.bufferD["clip_encoder"]]
            # self.clip_context.execute_v2(buffer_cilp_D)

            # buffer_cilp_D1 = [uncond_tokens.data_ptr(),self.ddim_sampler.bufferD["clip_encoder1"]]
            # self.clip_context.execute_v2(buffer_cilp_D1)

            # cond = {"c_concat": [control], "c_crossattn": [self.ddim_sampler.bufferD["clip_encoder"]]}
            # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.ddim_sampler.bufferD["clip_encoder1"]]}

            # cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            # -----clip的trt调用
            cond_text = [prompt + ', ' + a_prompt, n_prompt]
            cond_batch_encoding = self.tokenizer(cond_text, truncation=True, max_length=77, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            cond_tokens = cond_batch_encoding["input_ids"].type(torch.int32).to(self.device) #<---关键在这！
            
            #-------------替换为异步
            # buffer_cilp_D = [cond_tokens.data_ptr(),self.ddim_sampler.bufferD["clip_encoder"]]
            # self.clip_context.execute_v2(buffer_cilp_D)
            text_embeddings = runEngine(self.clip_engine, {"input_ids": cond_tokens},self.ddim_sampler.stream)['clip_encoder'].clone()

            shape = (4, H // 8, W // 8)

            # cond = self.model.get_learned_conditioning([prompt + ', ' + a_prompt, n_prompt])
            control = torch.cat([control],1)
            control = torch.cat([control,control],axis=0)

            # cond = {"c_concat": [control], "c_crossattn": [cond]}
            cond = {"c_concat": [control], "c_crossattn": [text_embeddings]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": None}

            # if config.save_memory:
            #     self.model.low_vram_shift(is_diffusing=True)

            # self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            # # vae
            # a = time.time()
            # buffer_vae_D = [samples.data_ptr(),self.ddim_sampler.bufferD["dec"]]
            # self.vae_context.execute_v2(buffer_vae_D)
            # cudart.cudaMemcpy(self.ddim_sampler.bufferH["dec"].ctypes.data, self.ddim_sampler.bufferD["dec"],
            #     self.ddim_sampler.bufferH["dec"].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            # results =  [self.ddim_sampler.bufferH["dec"].transpose((0,2,3,1))[i] for i in range(num_samples)]
            # b = time.time()
            # print((b-a)*1000)

            # vae cuda stream 
            buffer_vae = runEngine(self.vae_engine, {"image": samples},self.ddim_sampler.stream)['dec'].clone()
            results = [einops.rearrange(buffer_vae, 'b c h w -> b h w c') .cpu().numpy()[0]]

            
            
        return results