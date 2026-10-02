import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ==================== Shared building blocks ====================

from base.eeg_encoder import AdaptiveWaveletLayer, ResidualAdd


class InformationBottleneckLayer(nn.Module):
    """Combine global band priors with instance-specific energy-based offsets."""
    def __init__(self, n_scales: int = 5, init_temperature: float = 1.0):
        super().__init__()
        self.n_scales = n_scales
        self.eps = 1e-8

        # Global prior logits
        # Initialize from prior band importance
        init_mask_logits = torch.tensor([0.8, 1.0, 1.5, 1.2, 0.7])
        self.base_logits = nn.Parameter(init_mask_logits[:n_scales])

        # Predict instance-specific logit offsets from band energies
        self.adaptive_refiner = nn.Sequential(
            nn.Linear(n_scales, n_scales * 2),
            nn.ReLU(),
            nn.Linear(n_scales * 2, n_scales),
            nn.Tanh() # Bound offsets to [-1, 1] to preserve the prior
        )

        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, coeffs: List[torch.Tensor], compute_ib_loss: bool = True):
        # 1. Compute band-energy features [B, n_scales]
        band_energies = []
        for c in coeffs:
            energy = torch.mean(c**2, dim=(1, 2)) # [B]
            band_energies.append(torch.log(energy + self.eps))
        band_features = torch.stack(band_energies, dim=1)

        # 2. Combine global and adaptive logits
        adaptive_offset = self.adaptive_refiner(band_features)
        combined_logits = self.base_logits.unsqueeze(0) + adaptive_offset

        # 3. Compute band-selection weights
        temp = F.softplus(self.temperature).clamp(0.01, 10.0)
        masks = F.softmax(combined_logits / temp, dim=1) # [B, n_scales]

        # 4. Apply band gates
        gated_coeffs = [coeffs[i] * masks[:, i:i+1, None] for i in range(self.n_scales)]

        ib_loss = None
        if compute_ib_loss:
            # Minimize mask entropy to select relevant bands
            ib_loss = -torch.mean(torch.sum(masks * torch.log(masks + self.eps), dim=1))

        return gated_coeffs, ib_loss, masks


class WaveletInformationBottleneckDenoiser(nn.Module):
    def __init__(self, n_channels: int = 63, signal_len: int = 250, ib_beta: float = 0.01, scales: Optional[List[int]] = None, learnable_scales: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.signal_len = signal_len
        self.ib_beta = ib_beta
        self.eps = 1e-8

        self.wavelet = AdaptiveWaveletLayer(signal_len=signal_len, n_channels=n_channels, scales=scales, learnable_scales=learnable_scales)
        self.ib_layer = InformationBottleneckLayer(n_scales=self.wavelet.n_scales, init_temperature=1.0)

    def forward(self, x: torch.Tensor, return_info: bool = False) -> Tuple[torch.Tensor, dict]:
        # Wavelet decomposition
        coeffs = self.wavelet(x)
        gated_coeffs, ib_loss, masks = self.ib_layer(coeffs, compute_ib_loss=self.training)
        x_denoised = sum(gated_coeffs)

        if not return_info:
            return x_denoised, {}

        info = {
            'effective_scales': self.wavelet.get_effective_scales(),
            'center_frequencies': self.wavelet.get_center_frequencies(),
            'ib_masks': masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else None,
            'ib_loss': ib_loss if ib_loss is not None else torch.tensor(0.0, device=x.device)
        }
        return x_denoised, info

    def compute_ib_loss(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.wavelet(x)
        _, ib_loss, _ = self.ib_layer(coeffs, compute_ib_loss=True)
        return self.ib_beta * ib_loss if ib_loss is not None else torch.tensor(0.0, device=x.device)


# ==================== Projection layer ====================

class EEGProjectLayer(nn.Module):
    """Project optionally denoised neural signals into the CLIP embedding space."""

    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: Tuple[int, int] = (0, 250),
        drop_proj: float = 0.3,
        use_denoising: bool = True,
        ib_beta: float = 0.01,
        learnable_scales: bool = True
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.use_denoising = use_denoising

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        signal_len = self.timesteps[1] - self.timesteps[0]

        if use_denoising:
            self.denoiser = WaveletInformationBottleneckDenoiser(
                n_channels=c_num,
                signal_len=signal_len,
                ib_beta=ib_beta,
                learnable_scales=learnable_scales
            )

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(z_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_denoise_info: bool = False):
        if x.dim() == 2:
            x = x.view(x.shape[0], self.c_num, -1)

        denoise_info = {}
        ib_loss = torch.tensor(0.0, device=x.device)

        if self.use_denoising:
            x_clean, denoise_info = self.denoiser(x, return_info=return_denoise_info)
            if return_denoise_info:
                ib_loss = self.denoiser.compute_ib_loss(x)
        else:
            x_clean = x

        x_flat = x_clean.view(x_clean.shape[0], self.input_dim)
        embedding = self.model(x_flat)

        if return_denoise_info:
            denoise_info['ib_loss'] = ib_loss
            return embedding, denoise_info
        return embedding
