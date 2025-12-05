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


class ZigzagReorder(nn.Module):
    """Zigzag"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        #  x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 
        indices = self._get_zigzag_indices(H, W)
        
        # 
        x_flat = x.reshape(B, C, -1)
        x_reordered = torch.zeros_like(x_flat)
        
        for i, idx in enumerate(indices):
            x_reordered[:, :, i] = x_flat[:, :, idx]
        
        return x_reordered.reshape(B, C, H, W)
    
    def _get_zigzag_indices(self, h, w):
       
        indices = []
        for s in range(h + w - 1):
            if s % 2 == 0:  # 
                for i in range(min(s, h - 1), max(0, s - w + 1) - 1, -1):
                    j = s - i
                    if j < w:
                        indices.append(i * w + j)
            else:  # 
                for i in range(max(0, s - w + 1), min(s, h - 1) + 1):
                    j = s - i
                    if j < w:
                        indices.append(i * w + j)
        return indices


# fourier_modules.py
class FrequencyAlignModule(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fft = FFTLayer()
        self.ifft = IFFTLayer()
        self.align_net = None
        self.input_dim = None

        # 
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_ir, x_vis):
        # 
        f_ir = self.fft(x_ir)  # [B, H, W, C, 2]
        f_vis = self.fft(x_vis)

        B, H, W, C, _ = f_ir.shape

        
        if C != self.embed_dim:
            
            f_ir_real = f_ir[..., 0]  # [B, H, W, C]
            f_ir_imag = f_ir[..., 1]  # [B, H, W, C]

            f_vis_real = f_vis[..., 0]
            f_vis_imag = f_vis[..., 1]

            
            if not hasattr(self, 'channel_adjust_real'):
                self.channel_adjust_real = nn.Conv2d(C, self.embed_dim, 1).to(x_ir.device)
                self.channel_adjust_imag = nn.Conv2d(C, self.embed_dim, 1).to(x_ir.device)

           
            f_ir_real = self.channel_adjust_real(f_ir_real.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            f_ir_imag = self.channel_adjust_imag(f_ir_imag.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            f_vis_real = self.channel_adjust_real(f_vis_real.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            f_vis_imag = self.channel_adjust_imag(f_vis_imag.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            
            f_ir = torch.stack([f_ir_real, f_ir_imag], dim=-1)
            f_vis = torch.stack([f_vis_real, f_vis_imag], dim=-1)

            C = self.embed_dim

       
        f_ir_flat = f_ir.reshape(B, H * W, C * 2)
        f_vis_flat = f_vis.reshape(B, H * W, C * 2)

       
        combined = torch.cat([f_ir_flat, f_vis_flat], dim=-1)
        input_dim = combined.shape[-1]

        
        if self.align_net is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            self.align_net = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, C * 2)
            ).to(combined.device)

       
        f_aligned = self.align_net(combined)
        f_aligned = f_aligned.reshape(B, H, W, C, 2)

        
        aligned_feat = self.ifft(f_aligned)

       
        aligned_feat_permuted = aligned_feat.permute(0, 3, 1, 2)
        attn = self.attention(aligned_feat_permuted)
        aligned_feat = aligned_feat * attn.permute(0, 2, 3, 1)

        return aligned_feat



class FourierMambaScan(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # SS2D
        from models.cross import SS2D
        
        # 
        self.scan_modules = nn.ModuleList([
            SS2D(d_model=dim, d_state=d_state) 
            for _ in range(4)
        ])
        
        # zigzag
        self.zigzag = ZigzagReorder()
        
        # 
        self.activation = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C, H, W] 
        B, C, H, W = x.shape
        
        #  (zigzag)
        x_reordered = self.zigzag(x)
        
        # 
        scan_results = []
        for i, scan in enumerate(self.scan_modules):
            #  [B, H, W, C]
            scan_input = x_reordered.permute(0, 2, 3, 1)
            scan_result = scan(scan_input)
            scan_results.append(scan_result.permute(0, 3, 1, 2))
        
        #
        merged = sum(scan_results) / len(scan_results)
        
        #
        output = self.activation(merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return output

class DualBranchFusionBlock(nn.Module):
    def __init__(self, dim, d_state=16, step_size=2):
        super().__init__()
        self.dim = dim
        
        # CrossMambaBlock
        from models.cross import CrossMambaBlock
        
        # CroMB
        self.spatial_branch = CrossMambaBlock(
            d_model=dim, 
            d_state=d_state
        )
        
        #
        self.fourier_branch = nn.Sequential(
            FFTLayer(),
            FourierMambaScan(dim),  #
            IFFTLayer(),
            nn.Conv2d(dim, dim, 1)  #
        )
        
        # 
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 
        spatial_feat = self.spatial_branch(x1, x2)
        
        #
        fourier_input = x1 + x2
        fourier_feat = self.fourier_branch(fourier_input.permute(0, 3, 1, 2))
        fourier_feat = fourier_feat.permute(0, 2, 3, 1)
        
        
        gate_weights = self.gate(
            torch.cat([spatial_feat, fourier_feat], dim=-1).permute(0, 3, 1, 2)
        )
        gate_weights = gate_weights.permute(0, 2, 3, 1)
        
        gate_spatial, gate_fourier = torch.chunk(gate_weights, 2, dim=-1)
        
       
        fused_feat = spatial_feat * gate_spatial + fourier_feat * gate_fourier
        
        return fused_feat