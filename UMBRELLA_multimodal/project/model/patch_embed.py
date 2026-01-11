"""
UMBRELLA: Patch Embedding Module for 3D/4D Brain MRI

This module implements the PatchEmbed class for converting 3D T1-weighted MRI
and 4D fMRI volumes into patch embeddings for the LLaVA-based architecture.

Supports:
- sMRI: 3D structural MRI (original UMBRELLA)
- fMRI: 4D functional MRI (original UMBRELLA)
- T1: 3D T1-weighted MRI (BrainVLM style multimodal)
- FA: 3D Fractional Anisotropy from DTI (BrainVLM style multimodal)
- T1_FA: Late fusion of T1 and FA (two separate projections)
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for 3D sMRI and 4D fMRI volumes.

    This module converts volumetric brain images into sequences of patch embeddings
    suitable for transformer-based processing. It supports both:
    - 3D T1-weighted structural MRI (shape: B x C x D x H x W)
    - 4D fMRI functional MRI (shape: B x C x D x H x W x T)

    Args:
        sMRI_size: Size of sMRI images [D, H, W]
        sMRI_patch_size: Patch size for sMRI images [pD, pH, pW]
        fMRI_size: Size of fMRI volumes [D, H, W, T]
        fMRI_patch_size: Patch size for fMRI [pD, pH, pW, pT]
        T1_size: Size of T1 images [D, H, W] (for multimodal)
        T1_patch_size: Patch size for T1 [pD, pH, pW]
        FA_size: Size of FA images [D, H, W] (for multimodal)
        FA_patch_size: Patch size for FA [pD, pH, pW]
        in_chans: Number of input channels (default: 1)
        embed_dim: Embedding dimension (default: 768)
        dtype: Data type for parameters
        joint_optimization: If True, assert sMRI and fMRI have same patch count
        modality_type: "sMRI", "fMRI", "T1", "FA", or "T1_FA"
    """

    def __init__(self,
                 sMRI_size=[128, 128, 128],
                 sMRI_patch_size=[18, 18, 18],
                 fMRI_size=[96, 96, 96, 20],
                 fMRI_patch_size=[16, 16, 16, 5],
                 T1_size=[120, 120, 120],
                 T1_patch_size=[10, 10, 10],
                 FA_size=[120, 120, 120],
                 FA_patch_size=[10, 10, 10],
                 in_chans=1,
                 embed_dim=768,
                 dtype=torch.float32,
                 joint_optimization=False,
                 modality_type="sMRI"):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_type = modality_type

        # ============================================================
        # Patchifying layer for sMRI images (original UMBRELLA)
        # ============================================================
        sMRI_num_patches = (sMRI_size[0] // sMRI_patch_size[0]) * (sMRI_size[1] // sMRI_patch_size[1]) * (sMRI_size[2] // sMRI_patch_size[2])
        self.sMRI_grid_size = (sMRI_size[0] // sMRI_patch_size[0], sMRI_size[1] // sMRI_patch_size[1], sMRI_size[2] // sMRI_patch_size[2])
        self.sMRI_size = sMRI_size
        self.sMRI_patch_size = sMRI_patch_size
        self.sMRI_num_patches = sMRI_num_patches
        self.sMRI_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=sMRI_patch_size, stride=sMRI_patch_size, dtype=dtype)
        self.sMRI_positional_embeddings = nn.Parameter(torch.zeros(1, sMRI_num_patches, embed_dim))
        trunc_normal_(self.sMRI_positional_embeddings, std=.02)

        # ============================================================
        # Patchifying layer for fMRI (original UMBRELLA)
        # ============================================================
        fMRI_num_patches = (fMRI_size[0] // fMRI_patch_size[0]) * (fMRI_size[1] // fMRI_patch_size[1]) * (fMRI_size[2] // fMRI_patch_size[2]) * (fMRI_size[3] // fMRI_patch_size[3])
        self.fMRI_grid_size = (fMRI_size[0] // fMRI_patch_size[0], fMRI_size[1] // fMRI_patch_size[1], fMRI_size[2] // fMRI_patch_size[2], fMRI_size[3] // fMRI_patch_size[3])
        self.fMRI_size = fMRI_size
        self.fMRI_patch_size = fMRI_patch_size
        self.fMRI_num_patches = fMRI_num_patches
        self.fMRI_proj = nn.Linear(in_features=in_chans * fMRI_patch_size[0] * fMRI_patch_size[1] * fMRI_patch_size[2] * fMRI_patch_size[3], out_features=embed_dim)
        self.fMRI_positional_embeddings = nn.Parameter(torch.zeros(1, fMRI_num_patches, embed_dim))
        trunc_normal_(self.fMRI_positional_embeddings, std=.02)

        # ============================================================
        # Patchifying layer for T1 (BrainVLM style multimodal)
        # ============================================================
        T1_num_patches = (T1_size[0] // T1_patch_size[0]) * (T1_size[1] // T1_patch_size[1]) * (T1_size[2] // T1_patch_size[2])
        self.T1_grid_size = (T1_size[0] // T1_patch_size[0], T1_size[1] // T1_patch_size[1], T1_size[2] // T1_patch_size[2])
        self.T1_size = T1_size
        self.T1_patch_size = T1_patch_size
        self.T1_num_patches = T1_num_patches
        self.T1_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=T1_patch_size, stride=T1_patch_size, dtype=dtype)
        self.T1_positional_embeddings = nn.Parameter(torch.zeros(1, T1_num_patches, embed_dim))
        trunc_normal_(self.T1_positional_embeddings, std=.02)

        # ============================================================
        # Patchifying layer for FA (BrainVLM style multimodal)
        # ============================================================
        FA_num_patches = (FA_size[0] // FA_patch_size[0]) * (FA_size[1] // FA_patch_size[1]) * (FA_size[2] // FA_patch_size[2])
        self.FA_grid_size = (FA_size[0] // FA_patch_size[0], FA_size[1] // FA_patch_size[1], FA_size[2] // FA_patch_size[2])
        self.FA_size = FA_size
        self.FA_patch_size = FA_patch_size
        self.FA_num_patches = FA_num_patches
        self.FA_proj = nn.Conv3d(in_chans, embed_dim, kernel_size=FA_patch_size, stride=FA_patch_size, dtype=dtype)
        self.FA_positional_embeddings = nn.Parameter(torch.zeros(1, FA_num_patches, embed_dim))
        trunc_normal_(self.FA_positional_embeddings, std=.02)

        if joint_optimization:
            assert self.sMRI_num_patches == self.fMRI_num_patches, \
                f"For joint optimization, sMRI patches ({self.sMRI_num_patches}) must equal fMRI patches ({self.fMRI_num_patches})"

    def forward_sMRI(self, x):
        """Process sMRI image through patch embedding."""
        B, C, D, H, W = x.shape
        assert D == self.sMRI_size[0] and H == self.sMRI_size[1] and W == self.sMRI_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.sMRI_size[0]}*{self.sMRI_size[1]}*{self.sMRI_size[2]})."
        x = self.sMRI_proj(x).flatten(2).transpose(1, 2)  # B L C
        x = x + self.sMRI_positional_embeddings
        return x

    def forward_fMRI(self, x):
        """Process fMRI volume through patch embedding."""
        B, C, D, H, W, T = x.shape
        pD, pH, pW, pT = self.fMRI_grid_size
        sD, sH, sW, sT = self.fMRI_patch_size

        x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sD * sH * sW * sT * C)
        x = self.fMRI_proj(x)
        x = x.view(B, pD, pH, pW, -1, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        x = x.flatten(2).transpose(1, 2)  # B L C
        x = x + self.fMRI_positional_embeddings
        return x

    def forward_T1(self, x):
        """Process T1 image through patch embedding (BrainVLM style)."""
        B, C, D, H, W = x.shape
        assert D == self.T1_size[0] and H == self.T1_size[1] and W == self.T1_size[2], \
            f"T1 input size ({D}*{H}*{W}) doesn't match model ({self.T1_size[0]}*{self.T1_size[1]}*{self.T1_size[2]})."
        x = self.T1_proj(x).flatten(2).transpose(1, 2)  # B L C
        x = x + self.T1_positional_embeddings
        return x

    def forward_FA(self, x):
        """Process FA image through patch embedding (BrainVLM style)."""
        B, C, D, H, W = x.shape
        assert D == self.FA_size[0] and H == self.FA_size[1] and W == self.FA_size[2], \
            f"FA input size ({D}*{H}*{W}) doesn't match model ({self.FA_size[0]}*{self.FA_size[1]}*{self.FA_size[2]})."
        x = self.FA_proj(x).flatten(2).transpose(1, 2)  # B L C
        x = x + self.FA_positional_embeddings
        return x

    def forward_embeddings(self, x, modality=None):
        """Process a single modality input into patch embeddings.

        Args:
            x: Input tensor, either 5D (sMRI/T1/FA) or 6D (fMRI)
            modality: Optional modality hint ("sMRI", "fMRI", "T1", "FA", "T1_FA")

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # Use instance modality_type if not specified
        if modality is None:
            modality = self.modality_type

        if modality == "T1":
            return self.forward_T1(x)
        elif modality == "FA":
            return self.forward_FA(x)
        elif modality == "T1_FA":
            # T1_FA late fusion: batch is interleaved [S1_T1, S1_FA, S2_T1, S2_FA, ...]
            T1_images = x[0::2]  # T1 images (even indices)
            FA_images = x[1::2]  # FA images (odd indices)

            T1_embeddings = self.forward_T1(T1_images)
            FA_embeddings = self.forward_FA(FA_images)

            # Restore original interleaved order
            n_samples = T1_embeddings.shape[0]
            interleaved = torch.empty(
                (n_samples * 2, T1_embeddings.shape[1], T1_embeddings.shape[2]),
                device=x.device,
                dtype=T1_embeddings.dtype
            )
            interleaved[0::2] = T1_embeddings
            interleaved[1::2] = FA_embeddings
            return interleaved
        elif len(x.shape) == 5:
            # Default 5D: sMRI
            return self.forward_sMRI(x)
        elif len(x.shape) == 6:
            # Default 6D: fMRI
            return self.forward_fMRI(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    def forward(self, x, interpolate_pos_encoding=False, modality=None):
        """Forward pass for patch embedding.

        Args:
            x: Input tensor or dict of tensors keyed by modality
            interpolate_pos_encoding: Unused, for API compatibility
            modality: Optional modality hint

        Returns:
            Patch embeddings tensor
        """
        if isinstance(x, dict):
            modalities = list(x.keys())
            outputs = []
            for mod in modalities:
                # Handle 6D input from collator: [B, num_images, C, D, H, W]
                mod_data = x[mod]
                if len(mod_data.shape) == 6:
                    B, N, C, D, H, W = mod_data.shape
                    mod_data = mod_data.reshape(B * N, C, D, H, W)
                embeddings = self.forward_embeddings(mod_data, modality=mod)
                outputs.append(embeddings)
            outputs = torch.cat(outputs, dim=0)
        else:
            # Handle 6D input from collator: [B, num_images, C, D, H, W]
            if len(x.shape) == 6:
                B, N, C, D, H, W = x.shape
                x = x.reshape(B * N, C, D, H, W)
            outputs = self.forward_embeddings(x, modality=modality)

        return outputs

    def get_num_patches(self, modality=None):
        """Return the number of patches per image for given modality."""
        if modality is None:
            modality = self.modality_type
        if modality == "T1":
            return self.T1_num_patches
        elif modality == "FA":
            return self.FA_num_patches
        elif modality == "fMRI":
            return self.fMRI_num_patches
        else:
            return self.sMRI_num_patches
