'''

DDIM中的后处理算子抽离为pytorch模型，并序列化trt进行加速！
'''

import torch
import torch.nn as nn
import numpy as np


class PostNet(nn.Module):
    def __init__(self):
        super().__init__()

        # step = 20
        # self.alphas = torch.from_numpy(np.array([0.9983, 0.9505, 0.8930, 0.8264, 0.7521, 0.6722, 0.5888, 0.5048, 0.4229,0.3456, 0.2750, 
        #     0.2128, 0.1598, 0.1163, 0.0819, 0.0557, 0.0365, 0.0231,0.0140, 0.0082]))
        # self.alphas_prev = torch.from_numpy(np.array([0.99914998,0.99829602, 0.95052433, 0.89298052, 0.82639927, 0.75214338,
        #                     0.67215145, 0.58881873, 0.50481856, 0.42288151, 0.34555823, 0.27499905,
        #                     0.21278252, 0.15981644, 0.11632485, 0.08191671, 0.05571903, 0.03654652,
        #                     0.02307699, 0.0140049 ]))
        # self.sqrt_one_minus_alphas = torch.from_numpy(np.array([0.0413, 0.2224, 0.3271, 0.4167, 0.4979, 0.5726, 0.6412, 0.7037, 0.7597,
        #                                 0.8090, 0.8515, 0.8873, 0.9166, 0.9400, 0.9582, 0.9717, 0.9816, 0.9884,
        #                                 0.9930, 0.9959]))
        # self.sigmas = torch.from_numpy(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

        # self.time_range = [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101,51, 1]

        # step = 10

        self.alphas = torch.from_numpy(np.array([0.9983, 0.8930, 0.7521, 0.5888, 0.4229, 0.2750, 0.1598, 0.0819, 0.0365,0.0140]))
        self.alphas_prev = torch.from_numpy(np.array([0.99914998, 0.99829602, 0.89298052, 0.75214338, 0.58881873, 0.42288151,0.27499905,  0.15981644, 0.08191671, 0.03654652]))
        self.sqrt_one_minus_alphas = torch.from_numpy(np.array([0.0413, 0.3271, 0.4979, 0.6412, 0.7597, 0.8515, 0.9166, 0.9582, 0.9816,
                0.9930]))
        self.sigmas = torch.from_numpy(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

    def forward(self,x,image,unconditional_guidance_scale,index):
        e_t = image[1].unsqueeze(0) + unconditional_guidance_scale * (image[0].unsqueeze(0) - image[1].unsqueeze(0))


        a_t = self.alphas[index]
        a_prev =  self.alphas_prev[index]
        # sigma_t = self.sigmas[index] #因为sigma_t=0
        sqrt_one_minus_at = self.sqrt_one_minus_alphas[index]

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        # dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t #因为sigma_t=0
        dir_xt = (1. - a_prev).sqrt() * e_t

        # noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature  #tem=1
        
        # 再减小运算，因为sigma=0
        # noise = sigma_t * torch.randn(x.shape)
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt

        return x_prev,pred_x0


if __name__ == "__main__":

    postnet = PostNet()
    postnet = postnet.eval()

    torch.save(postnet,"./models/postnet.pth")

    # 转onnx
    onnx_path = "./models/postnet.onnx"
    input_names =["x","image","unconditional_guidance_scale","index"]
    output_names = ['x_prev',"pred_x0"]


    dummy_input = (torch.randn((1,4,32,48)),
        torch.randn((2,4,32,48)),
        torch.randn([1]),
        1,
        )
    # dummy_input = torch.randn(1, 3, 256, 256)

    # torch.onnx.export(model,dummy_input, onnx_path, verbose=True)
    torch.onnx.export(postnet, dummy_input, onnx_path, verbose=True,input_names=input_names,output_names=output_names,
                opset_version=18
                    )
    print("postnet torch2onnx success!!!")