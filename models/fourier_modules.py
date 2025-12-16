import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class FFTLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        is_channels_last = x.dim() == 4 and x.shape[-1] < x.shape[-2]

        if is_channels_last:
            # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)

        
        H, W = x.shape[2], x.shape[3]
        H_pad = 2 ** math.ceil(math.log2(H))
        W_pad = 2 ** math.ceil(math.log2(W))
        pad_h = (H_pad - H) // 2
        pad_w = (W_pad - W) // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        try:
            
            x_freq = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        except RuntimeError as e:
            #print(f"CUFFT error, falling back to CPU: {e}")
            x_cpu = x.cpu()
            x_freq = torch.fft.fft2(x_cpu, dim=(-2, -1), norm='ortho').to(x.device)

        #  [B, C, H, W, 2]
        x_freq = torch.stack([x_freq.real, x_freq.imag], dim=-1)

        if is_channels_last:
            # [B, C, H, W, 2] -> [B, H, W, C, 2]
            x_freq = x_freq.permute(0, 2, 3, 1, 4)

        return x_freq


class IFFTLayer(nn.Module):
    

    def __init__(self):
        super().__init__()

    def forward(self, x_freq):
        #  x_freq: [B, C, H, W, 2]  [B, H, W, C, 2]
        is_channels_last = x_freq.dim() == 5 and x_freq.shape[-2] < x_freq.shape[-3]

        if is_channels_last:
            # [B, H, W, C, 2] -> [B, C, H, W, 2]
            x_freq = x_freq.permute(0, 3, 1, 2, 4)

      
        x_freq_complex = torch.complex(x_freq[..., 0], x_freq[..., 1])

        try:
          
            x_spatial = torch.fft.ifft2(x_freq_complex, dim=(-2, -1), norm='ortho').real
        except RuntimeError as e:
            #print(f"CUFFT error, falling back to CPU: {e}")
            x_freq_complex_cpu = x_freq_complex.cpu()
            x_spatial = torch.fft.ifft2(x_freq_complex_cpu, dim=(-2, -1), norm='ortho').real.to(x_freq_complex.device)


       

        if is_channels_last:
            # [B, C, H, W] -> [B, H, W, C]
            x_spatial = x_spatial.permute(0, 2, 3, 1)

        return x_spatial


