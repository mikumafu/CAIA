import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# Information-Bottleneck Wavelet EEG Encoder

class ResidualAdd(nn.Module):
    """Add a residual branch to its input."""
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class AdaptiveWaveletLayer(nn.Module):
    """Decompose neural signals with differentiable Morlet wavelet scales."""

    def __init__(
        self,
        signal_len: int = 250,
        n_channels: int = 63,
        scales: Optional[List[int]] = None,
        learnable_scales: bool = True
    ):
        super().__init__()
        self.signal_len = signal_len
        self.n_channels = n_channels
        self.learnable_scales = learnable_scales

        # Morlet wavelet parameters
        self.omega0 = 6.0
        self.eps = 1e-8

        if scales is None:
            base_scales = [62, 31, 19, 8, 5]  # Delta → Gamma
        else:
            base_scales = scales

        self.n_scales = len(base_scales)

        self.register_buffer('base_scales', torch.tensor(base_scales, dtype=torch.float32))

        if learnable_scales:
            init_factors = torch.tensor([1.2, 1.1, 1.0, 0.9, 0.8])
            self.scale_factors = nn.Parameter(init_factors[:self.n_scales])
        else:
            self.register_buffer('scale_factors', torch.ones(self.n_scales))

    def _generate_morlet_kernel_differentiable(
        self,
        scale: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:

        # Compute the integer kernel length outside autograd
        scale_value = scale.item()
        kernel_len = min(int(scale_value * 4), self.signal_len // 2)
        kernel_len = max(kernel_len, 3)  # Enforce the minimum kernel length

        # Build the time axis
        half_len = kernel_len // 2
        t = torch.arange(-half_len, half_len + 1, device=device, dtype=torch.float32)

        # Compute the Morlet wavelet with differentiable scale
        sigma = scale / 2.5
        sigma_sq = 2 * sigma ** 2 + self.eps

        # Gaussian window
        gaussian = torch.exp(-t ** 2 / sigma_sq)

        # Oscillatory component
        oscillation = torch.cos(self.omega0 * t / (scale + self.eps))

        # Morlet wavelet
        kernel = gaussian * oscillation

        # L2 normalization
        kernel_norm = torch.norm(kernel) + self.eps
        kernel = kernel / kernel_norm

        return kernel

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Transform the input tensor and return the computed features."""
        batch, channels, time = x.shape
        coeffs = []

        # Compute effective scales
        if self.learnable_scales:
            scale_factors = F.softplus(self.scale_factors).clamp(0.5, 2.0)
        else:
            scale_factors = self.scale_factors

        effective_scales = self.base_scales * scale_factors

        x_reshaped = x.view(batch * channels, 1, time)

        # Apply the wavelet transform at each scale
        for scale_idx in range(self.n_scales):
            scale = effective_scales[scale_idx]

            # Generate the wavelet kernel
            kernel = self._generate_morlet_kernel_differentiable(
                scale=scale,
                device=x.device
            )

            # Reshape: [1, 1, kernel_len]
            kernel = kernel.unsqueeze(0).unsqueeze(0)

            # Batched convolution
            pad_len = kernel.shape[-1] // 2

            wt_coeff = F.conv1d(
                x_reshaped,
                kernel,
                padding=pad_len
            )

            # Crop to the original signal length
            if wt_coeff.shape[-1] > time:
                start = (wt_coeff.shape[-1] - time) // 2
                wt_coeff = wt_coeff[:, :, start:start + time]
            elif wt_coeff.shape[-1] < time:
                pad_needed = time - wt_coeff.shape[-1]
                wt_coeff = F.pad(wt_coeff, (pad_needed // 2, pad_needed - pad_needed // 2))

            coeffs.append(wt_coeff.view(batch, channels, time))

        return coeffs

    def get_effective_scales(self) -> List[float]:
        """Return the current effective wavelet scales."""
        if self.learnable_scales:
            factors = F.softplus(self.scale_factors).clamp(0.5, 2.0)
            factors = factors.detach().cpu().numpy()
        else:
            factors = self.scale_factors.cpu().numpy()

        base = self.base_scales.cpu().numpy()
        return (base * factors).tolist()

    def get_center_frequencies(self, sampling_rate: float = 250.0) -> List[float]:
        """Return the scale-to-frequency diagnostic values in Hz."""
        effective_scales = self.get_effective_scales()
        return [self.omega0 * sampling_rate / (2 * np.pi * s) for s in effective_scales]


class InformationBottleneckLayer(nn.Module):
    """Learn temperature-controlled frequency-band selection weights."""

    def __init__(self, n_scales: int = 5, init_temperature: float = 1.0):
        super().__init__()
        self.n_scales = n_scales
        self.eps = 1e-8

        init_mask_logits = torch.tensor([0.7, 0.9, 1.2, 1.0, 0.6])
        self.mask_logits = nn.Parameter(init_mask_logits[:n_scales])
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, coeffs: List[torch.Tensor], compute_ib_loss: bool = True):
        temp = F.softplus(self.temperature).clamp(0.01, 10.0)
        masks = F.softmax(self.mask_logits / temp, dim=0)
        gated_coeffs = [coeff * mask for coeff, mask in zip(coeffs, masks)]

        ib_loss = None
        if compute_ib_loss:
            ib_loss = -torch.sum(masks * torch.log(masks + self.eps))

        return gated_coeffs, ib_loss, masks


class WaveletInformationBottleneckDenoiser(nn.Module):
    """Combine wavelet decomposition and entropy-regularized band selection."""

    def __init__(
        self,
        n_channels: int = 63,
        signal_len: int = 250,
        ib_beta: float = 0.01,
        scales: Optional[List[int]] = None,
        learnable_scales: bool = True
    ):
        super().__init__()
        self.n_channels = n_channels
        self.signal_len = signal_len
        self.ib_beta = ib_beta
        self.eps = 1e-8

        self.wavelet = AdaptiveWaveletLayer(
            signal_len=signal_len,
            n_channels=n_channels,
            scales=scales,
            learnable_scales=learnable_scales
        )
        n_scales = self.wavelet.n_scales

        self.ib_layer = InformationBottleneckLayer(
            n_scales=n_scales,
            init_temperature=1.0
        )

        scale_importance = [0.8, 1.0, 1.5, 1.2, 0.7] # EEG
        init_weights = torch.tensor(scale_importance[:n_scales], dtype=torch.float32)
        self.fusion_weights = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor, return_info: bool = False) -> Tuple[torch.Tensor, dict]:
        coeffs = self.wavelet(x)
        gated_coeffs, ib_loss, masks = self.ib_layer(coeffs, compute_ib_loss=self.training)
        weights = F.softmax(self.fusion_weights, dim=0)
        x_denoised = sum(w * c for w, c in zip(weights, gated_coeffs))

        if not return_info:
            return x_denoised, {}

        info = {
            'effective_scales': self.wavelet.get_effective_scales(),
            'center_frequencies': self.wavelet.get_center_frequencies(),
            'ib_masks': masks.detach().cpu().numpy(),
            'ib_temperature': F.softplus(self.ib_layer.temperature).item(),
            'fusion_weights': weights.detach().cpu().numpy(),
            'ib_loss': ib_loss.item() if ib_loss is not None else 0.0
        }

        return x_denoised, info

    def compute_ib_loss(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.wavelet(x)
        _, ib_loss, _ = self.ib_layer(coeffs, compute_ib_loss=True)
        return self.ib_beta * ib_loss if ib_loss is not None else torch.tensor(0.0, device=x.device)


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
