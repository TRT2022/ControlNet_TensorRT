"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import time

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

from cuda import cudart
import tensorrt as trt
import ctypes

from trt_util import *
from polygraphy import cuda

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
# layernorm = ctypes.CDLL("./layerNormPlugin/layerNormKernel.so")
# layernorm = ctypes.CDLL("./LayerNormPlugin/LayerNormPlugin.so")
groupnorm = ctypes.CDLL("./groupNormPlugin/groupNormKernel.so")
# seqLen2Spatial = ctypes.CDLL("./seqLen2SpatialPlugin/seqLen2SpatialKernel.so")
# split_gelu = ctypes.CDLL("./splitGeLUPlugin/splitGeLUKernel.so")




def get_engine(src_file,dynamic=False):
    with open(src_file, 'rb') as planfile:
        engine = trt.Runtime(logger).deserialize_cuda_engine(planfile.read())
    context = engine.create_execution_context()

    # nIO=engine.num_io_tensors
    # ITensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    # print("----------------")
    # print(src_file)
    # print(ITensorName)
    # print("--------------------")
    if dynamic:
        context.set_binding_shape(0,[2,77])
    
    return context

def cumallocs():
    '''分配模型所用到的内存和显存'''
    # 内存申请
    bufferH = {
        "input_id":np.ascontiguousarray(np.arange(2*77,dtype=np.int64).reshape(2,77)),
        "clip_encoder":np.ascontiguousarray(np.arange(2*77*768,dtype=np.float32).reshape(2,77,768)),
        # "clip_encoder1":np.ascontiguousarray(np.arange(1*77*768,dtype=np.float32).reshape(1,77,768)),

        "image":np.ascontiguousarray(np.arange(1*4*32*48,dtype=np.float32).reshape(1,4,32,48)),
        "dec":np.ascontiguousarray(np.arange(1*3*256*384,dtype=np.float32).reshape(1,3,256,384)),
        "x_noisy":np.ascontiguousarray(np.arange(2*4*32*48,dtype=np.float32).reshape(2,4,32,48)),  # 这四个controlnet和Unet共享
        "hint":np.ascontiguousarray(np.arange(2*3*256*384,dtype=np.float32).reshape(2,3,256,384)),
        "timesteps":np.ascontiguousarray(np.arange(2,dtype=np.float32).reshape(2,)),
        #"context": 就是clip_encoder
        "control_0":np.ascontiguousarray(np.arange(2*320*32*48,dtype=np.float32).reshape(2, 320, 32, 48)), #controlnet的output也是unet的input
        "control_1":np.ascontiguousarray(np.arange(2*320*32*48,dtype=np.float32).reshape(2, 320, 32, 48)),
        "control_2":np.ascontiguousarray(np.arange(2*320*32*48,dtype=np.float32).reshape(2, 320, 32, 48)),
        "control_3":np.ascontiguousarray(np.arange(2*320*16*24,dtype=np.float32).reshape(2, 320, 16, 24)),
        "control_4":np.ascontiguousarray(np.arange(2*640*16*24,dtype=np.float32).reshape(2, 640, 16, 24)),
        "control_5":np.ascontiguousarray(np.arange(2*640*16*24,dtype=np.float32).reshape(2, 640, 16, 24)),
        "control_6":np.ascontiguousarray(np.arange(2*640*8*12,dtype=np.float32).reshape(2, 640, 8, 12)),
        "control_7":np.ascontiguousarray(np.arange(2*1280*8*12,dtype=np.float32).reshape(2, 1280, 8, 12)),
        "control_8":np.ascontiguousarray(np.arange(2*1280*8*12,dtype=np.float32).reshape(2, 1280, 8, 12)),
        "control_9":np.ascontiguousarray(np.arange(2*1280*4*6,dtype=np.float32).reshape(2, 1280, 4, 6)),
        "control_10":np.ascontiguousarray(np.arange(2*1280*4*6,dtype=np.float32).reshape(2, 1280, 4, 6)),
        "control_11":np.ascontiguousarray(np.arange(2*1280*4*6,dtype=np.float32).reshape(2, 1280, 4, 6)),
        "control_12":np.ascontiguousarray(np.arange(2*1280*4*6,dtype=np.float32).reshape(2, 1280, 4, 6)),
        #"unet_out":[1, 4, 32, 48]是vaedecoder的input image（经过了一些运算）

        # # postnet
        # "ugs":np.ascontiguousarray(np.arange(1,dtype=np.float32).reshape([1])),
        # "idx":np.ascontiguousarray(np.arange(1,dtype=np.int32).reshape([1]))

    }
    #显存分配
    bufferD = {}
    for key in bufferH.keys():
        bufferD[key] = cudart.cudaMalloc(bufferH[key].nbytes)[1]
    return bufferH,bufferD

def cumalfrees(bufferD):
    '''清空所有分配的模型IO显存'''
    for key in bufferD.keys():
        cudart.cudaFree(bufferD[key])
    return 0

class DDIMSampler(object):
    def __init__(self, model, combine_context, postnet_context,schedule="linear",**kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        # self.controlnet_context = controlnet_context
        # self.unet_context = unet_context
        self.combine_context = combine_context
        self.postnet_context = postnet_context

        # 申请所有显存
        # self.bufferH,self.bufferD = cumallocs()
        self.image = torch.zeros((2,4,32,48)).to(torch.device("cuda"))
        self.x_prev = torch.zeros((1,4,32,48)).to(torch.device("cuda"))
        self.pred_x0 = torch.zeros((1,4,32,48)).to(torch.device("cuda"))
        self.ugs = torch.zeros((1,)).to(torch.device("cuda"))
        self.indx = torch.zeros((1,)).to(torch.device("cuda"))

        self.indexs = [torch.tensor(index).type(torch.int32).to(torch.device("cuda")) for index in range(10)]  #20
        
        self.trange = [901, 801, 701, 601, 501, 401, 301, 201, 101,   1]#[951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101,51, 1]
        self.tss = [torch.full((1,), step, device=torch.device("cuda"), dtype=torch.int32) for step in self.trange]

        # cuda stream
        self.stream = cuda.Stream()
        # self.stream1 = cuda.Stream()

      
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        # intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        time_range = self.trange
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range[1:], desc='DDIM Sampler', total=total_steps) #<---------------
        self.ugs = torch.tensor(unconditional_guidance_scale).type(torch.float32).to(torch.device("cuda")) #<--postnet

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            # ts = torch.full((b,), step, device=device, dtype=torch.long)
            # ts = torch.full((2,), step, device=device, dtype=torch.long)  # <----------
            ts = self.tss[i]
       
            # if mask is not None:
            #     assert x0 is not None
            #     img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            #     img = img_orig * mask + (1. - mask) * img

            # if ucg_schedule is not None:
            #     assert len(ucg_schedule) == len(time_range)
            #     unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            # if callback: callback(i)
            # if img_callback: img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            #     intermediates['x_inter'].append(img)
            #     intermediates['pred_x0'].append(pred_x0)

        # return img, intermediates
        return img, None


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        #     model_output = self.model.apply_model(x, t, c)
        # else:
        # 修改一下
        # model_t = self.model.apply_model(x, t, c)
        # model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
        # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        
        # 将下面面的调用整成batch=2
        # # 调用p_sample_ddim # 计算conditon的
        # #step1 调用controlnet: 
        # hint = torch.cat(c['c_concat'],1)
        # buffer_controlnet_D = [x.data_ptr(),hint.data_ptr(),t.data_ptr(),c['c_crossattn'][0].data_ptr()]+[self.bufferD[f"control_{i}"] for i in range(13)]
        # self.controlnet_context.execute_v2(buffer_controlnet_D)
        # # step2 调用sd:
        # buffer_unet_D = [x.data_ptr(),t.data_ptr(),c['c_crossattn'][0].data_ptr()]+ [self.bufferD[f"control_{i}"] for i in range(13)]+[self.image.data_ptr()]
        # self.unet_context.execute_v2(buffer_unet_D)
        # #eps_c = diffusion_model(x=x_noisy, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control) # Unet
        # model_t = self.image.clone()
        # # 计算uncondation的
        # #step1: 
        # buffer_controlnet_D1 = [x.data_ptr(),hint.data_ptr(),t.data_ptr(),unconditional_conditioning['c_crossattn'][0].data_ptr()]+[self.bufferD[f"control_{i}"] for i in range(13)]
        # self.controlnet_context.execute_v2(buffer_controlnet_D1)
        # # step2:
        # buffer_unet_D1 = [x.data_ptr(),t.data_ptr(),unconditional_conditioning['c_crossattn'][0].data_ptr()] + [self.bufferD[f"control_{i}"] for i in range(13)]+[self.image.data_ptr()]
        # self.unet_context.execute_v2(buffer_unet_D1)

        # model_output = self.image + unconditional_guidance_scale * (model_t - self.image)

        
        # # 2. hint 在cond中做， x在这里做，t在这里做
        # aa = time.time()
        # x2 = torch.cat([x,x],axis=0)
        # bb = time.time()

        # print((bb-aa)*1000*20)
        # buffer_controlnet_D = [x2.data_ptr(),c["c_concat"][0].data_ptr(),t.data_ptr(),self.bufferD["clip_encoder"]]+[self.bufferD[f"control_{i}"] for i in range(13)]
        # self.controlnet_context.execute_v2(buffer_controlnet_D)
        # # step2 调用sd:
        # buffer_unet_D = [x2.data_ptr(),t.data_ptr(),self.bufferD["clip_encoder"]]+ [self.bufferD[f"control_{i}"] for i in range(13)]+[self.image.data_ptr()]
        # self.unet_context.execute_v2(buffer_unet_D)
        # #eps_c = diffusion_model(x=x_noisy, timesteps=ts, context=cond_txt, control=control, only_mid_control=self.only_mid_control) # Unet
        # # model_t = self.image[0].unsqueeze(0)

        #3. combine unet and controlnet

        x2 = torch.cat([x,x],axis=0)
        # # buffer_combine_D = [x2.data_ptr(),c["c_concat"][0].data_ptr(),t.data_ptr(),self.bufferD["clip_encoder"]]+[self.image.data_ptr()]
        # buffer_combine_D = [x2.data_ptr(),c["c_concat"][0].data_ptr(),t.data_ptr(),c["c_crossattn"][0].data_ptr()]+[self.image.data_ptr()]

        # self.combine_context.execute_v2(buffer_combine_D)
        #----combine cuda strem
        image1 = runEngine(self.combine_context, {"x_noisy": x2,"hint":c["c_concat"][0],"timesteps":t,\
            "context":c["c_crossattn"][0]},self.stream)["unet_out"].clone()
 
        # #---------------这部分后处理用TRT实现-----------------
        # aa = time.time()
        # e_t = self.image[1].unsqueeze(0) + unconditional_guidance_scale * (self.image[0].unsqueeze(0) -self.image[1].unsqueeze(0))

        # # if self.model.parameterization == "v": #self.model.parameterization=eps
        # #     e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        # # else:
        # #     e_t = model_output

        # # if score_corrector is not None:  #False 不会走
        # #     assert self.model.parameterization == "eps", 'not implemented'
        # #     e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        # alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        # sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

  

        # # select parameters corresponding to the currently considered timestep
        # a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        # a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        # sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # # current prediction for x_0
        # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # # if self.model.parameterization != "v":
        # #     pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # # else:
        # #     pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
            
        # # if quantize_denoised: #False
        # #     pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # # if dynamic_threshold is not None:
        # #     raise NotImplementedError()

        # # direction pointing to x_t
        # dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # # noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature  #tem=1
        # noise = sigma_t * noise_like(x.shape, device, repeat_noise)   #tem=1

        # # if noise_dropout > 0.:  #0
        # #     noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # bb = time.time()
        # # print((bb-aa)*1000*20)


        # TRT实现后处理 "alphas","alphas_prev","sqrt_one_minus_alphas","sigmas"
        self.indx = self.indexs[index]
        buffer_postnet_D = [x.data_ptr(),image1.data_ptr(),self.ugs.data_ptr(),self.indx.data_ptr()
            ]+[self.pred_x0.data_ptr(),self.x_prev.data_ptr()]
        self.postnet_context.execute_v2(buffer_postnet_D)

        # postnet cuda stream 反而更慢了！
        
        # postnet = runEngine(self.postnet_context, {"x": x,"image":image1,"unconditional_guidance_scale":self.ugs,\
        #     "index":self.indx},self.stream)
    
        # self.x_prev = postnet['x_prev'].clone()
        # self.pred_x0 = postnet["pred_x0"].clone()
        
        return self.x_prev, self.pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
