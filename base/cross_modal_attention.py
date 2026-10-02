"""
Cross-Modal Attention with Multiple Fusion Modes
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from scipy.stats import norm
import logging


class CrossModalAttention(nn.Module):
    """Fuse saliency- and center-blurred images using EEG queries.

    Supported modes: attention, learnable_mask, linear_weight, blur_a_only, blur_b_only."""
    def __init__(self, eeg_dim, img_channels=3, debug=False, fusion_mode='attention', img_size=(224, 224), gradient_checkpointing=True):
        super().__init__()

        self.debug = debug
        self.fusion_mode = fusion_mode
        self.img_size = img_size
        self.gradient_checkpointing = gradient_checkpointing

        # Downsampled feature size
        self.feature_size = (img_size[0] // 4, img_size[1] // 4)  # 56×56

        # ========== Cross-modal attention ==========
        if fusion_mode == 'attention':
            hidden_dim = 64

            logging.info("Initialize cross-modal attention")
            logging.info(f"  - Single-head attention (hidden_dim={hidden_dim})")
            logging.info(f"  - Feature resolution: {self.feature_size[0]}×{self.feature_size[1]}")
            logging.info("  - Normalization: InstanceNorm2d")
            logging.info("  - Activation: Sigmoid (Centering + Temperature scaling)")

            # ========== Learnable temperature ==========
            self.log_temperature = nn.Parameter(torch.zeros(1))
            logging.info(f"  - Initial temperature: {torch.exp(self.log_temperature).item():.2f}")

            # ========== EEG query branch ==========
            self.eeg_query = nn.Sequential(
                nn.Linear(eeg_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

            # ========== Image key branch==========
            self.img_key = nn.Sequential(
                # Layer 1: Standard convolution
                nn.Conv2d(img_channels * 2, 32, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(32),
                nn.GELU(),

                # Layer 2: Dilated convolution
                nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.InstanceNorm2d(32),
                nn.GELU(),

                # Layer 3: Downsample (224 → 112)
                nn.Conv2d(32, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.GELU(),

                # Layer 4: Downsample (112 → 56)
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.GELU(),
            )

            self._init_weights()

        # ========== Other fusion modes ==========
        elif fusion_mode == 'learnable_mask':
            H, W = img_size
            self.mask = nn.Parameter(torch.full((1, 1, H, W), 0.5))
            logging.info(f"Initialize learnable mask: shape={self.mask.shape}")

        elif fusion_mode == 'linear_weight':
            self.alpha = nn.Parameter(torch.tensor(0.5))
            logging.info("Initialize linear weighted fusion: alpha=0.5")

        elif fusion_mode in ['blur_a_only', 'blur_b_only']:
            logging.info(f"Initialize fusion mode: {fusion_mode}")

        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"CrossModalAttention initialized (mode: {fusion_mode}, Parameter count: {total_params:,})")

    def _init_weights(self):
        """Initialize projection, convolution, and normalization weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, eeg_features, blur_a, blur_b):
        B, C, H, W = blur_a.shape
        device = blur_a.device

        # ========== Cross-modal attention ==========
        if self.fusion_mode == 'attention':
            # ========== Step 1: Generate EEG queries ==========
            Q = self.eeg_query(eeg_features)  # [B, 64]
            Q = Q.unsqueeze(-1)  # [B, 64, 1]

            # ========== Step 2: Generate image keys ==========
            img_concat = torch.cat([blur_a, blur_b], dim=1)  # [B, 6, 224, 224]
            # Recompute stateless InstanceNorm/GELU blocks during backward to save memory.
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                shared_features = checkpoint(self.img_key, img_concat, use_reentrant=False)
            else:
                shared_features = self.img_key(img_concat)  # [B, 64, 56, 56]

            H_down, W_down = self.feature_size
            N_down = H_down * W_down

            K = shared_features.view(B, 64, N_down)  # [B, 64, 3136]

            # ========== Step 3: Compute attention scores ==========
            attn_scores = torch.bmm(Q.transpose(1, 2), K)  # [B, 1, 3136]

            if self.debug:
                print(f"[DEBUG] attn_scores (raw): [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

            # ========== Step 4: Centering ==========
            attn_scores_mean = attn_scores.mean(dim=-1, keepdim=True)  # [B, 1, 1]
            attn_scores_centered = attn_scores - attn_scores_mean       # [B, 1, 3136]

            if self.debug:
                print(f"[DEBUG] attn_scores_centered: [{attn_scores_centered.min():.4f}, {attn_scores_centered.max():.4f}]")
                print(f"[DEBUG] attn_scores_centered mean: {attn_scores_centered.mean():.6f} (should be close to zero)")

            # ========== Step 5: Temperature scaling ==========
            temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=5.0)
            attn_scores_scaled = attn_scores_centered / temperature

            if self.debug:
                print(f"[DEBUG] temperature: {temperature.item():.4f}")
                print(f"[DEBUG] attn_scores_scaled: [{attn_scores_scaled.min():.4f}, {attn_scores_scaled.max():.4f}]")

            # ========== Step 6: Sigmoid ==========
            attn_map_low = torch.sigmoid(attn_scores_scaled)  # [B, 1, 3136]

            if self.debug:
                print(f"[DEBUG] attn_map_low (sigmoid): [{attn_map_low.min():.6f}, {attn_map_low.max():.6f}]")
                print(f"[DEBUG] attn_map_low mean: {attn_map_low.mean():.6f}")
                print(f"[DEBUG] attn_map_low std: {attn_map_low.std():.6f}")

            # ========== Step 7: Reshape + Upsample ==========
            attn_map_low = attn_map_low.view(B, 1, H_down, W_down)
            attn_map = torch.nn.functional.interpolate(
                attn_map_low, size=(H, W), mode='bilinear', align_corners=False
            )

            if self.debug:
                print(f"[DEBUG] attn_map (final): [{attn_map.min():.6f}, {attn_map.max():.6f}]")
                print(f"[DEBUG] attn_map std: {attn_map.std():.6f}")

            # ========== Step 8: Spatial soft selection between images ==========
            fused_img = attn_map * blur_a + (1 - attn_map) * blur_b

            return fused_img, attn_map

        # ========== Other fusion modes ==========
        elif self.fusion_mode == 'learnable_mask':
            if self.mask.shape[2:] != (H, W):
                mask_resized = torch.nn.functional.interpolate(
                    self.mask, size=(H, W), mode='bilinear', align_corners=False
                )
                attn_map = torch.sigmoid(mask_resized)
            else:
                attn_map = torch.sigmoid(self.mask)

            fused_img = attn_map * blur_a + (1 - attn_map) * blur_b
            attn_map_batch = attn_map.expand(B, 1, H, W)

            return fused_img, attn_map_batch

        elif self.fusion_mode == 'linear_weight':
            alpha = torch.sigmoid(self.alpha)
            fused_img = alpha * blur_a + (1 - alpha) * blur_b
            attn_map = torch.full((B, 1, H, W), alpha.item(), device=device)

            return fused_img, attn_map

        elif self.fusion_mode == 'blur_a_only':
            fused_img = blur_a
            attn_map = torch.ones((B, 1, H, W), device=device)
            return fused_img, attn_map

        elif self.fusion_mode == 'blur_b_only':
            fused_img = blur_b
            attn_map = torch.zeros((B, 1, H, W), device=device)
            return fused_img, attn_map


class BoundaryDistanceLoss(nn.Module):
    """Penalize matched-pair similarities outside the running confidence interval."""
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.z_alpha_2 = norm.ppf(1 - alpha / 2)
        logging.info(f"Initialize BoundaryDistanceLoss (α={alpha}, z_α/2={self.z_alpha_2:.4f})")

    def forward(self, logits_per_image, batch_indices, running_stats):
        device = logits_per_image.device
        diagonal_sim = torch.diagonal(logits_per_image)

        mean = running_stats['mean']
        std = running_stats['std']

        lower_bound = torch.tensor(mean - self.z_alpha_2 * std, device=device)
        upper_bound = torch.tensor(mean + self.z_alpha_2 * std, device=device)

        distance_to_lower = torch.relu(lower_bound - diagonal_sim)
        distance_to_upper = torch.relu(diagonal_sim - upper_bound)
        boundary_distance = distance_to_lower + distance_to_upper

        with torch.no_grad():
            abnormal_mask = (diagonal_sim < lower_bound) | (diagonal_sim > upper_bound)
            num_abnormal = abnormal_mask.sum().item()

        if num_abnormal == 0:
            loss = boundary_distance.mean() * 0.0
        else:
            loss = boundary_distance[abnormal_mask].mean()

        return loss, abnormal_mask
