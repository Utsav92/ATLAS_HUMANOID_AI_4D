"""Multimodal Sensor Fusion Module for ATLAS HUMANOID AI 4D.
Fuses proprioception, vision, depth, tactile signals, and language instruction embeddings.
Supports missing-modality masks and compares Concatenation MLP with Cross-Attention.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionNet(nn.Module):
    """Fuses multi-sensory modalities into a unified robot decision representation."""
    def __init__(
        self,
        proprio_dim: int = 50,
        vision_dim: int = 32,
        tactile_dim: int = 16,
        language_dim: int = 16,
        fused_dim: int = 64,
        out_action_dim: int = 43
    ):
        super().__init__()
        # Encoders for each modality
        self.enc_proprio = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ReLU())
        self.enc_vision = nn.Sequential(nn.Linear(vision_dim, 32), nn.ReLU())
        self.enc_tactile = nn.Sequential(nn.Linear(tactile_dim, 16), nn.ReLU())
        self.enc_lang = nn.Sequential(nn.Linear(language_dim, 16), nn.ReLU())

        # Total concatenated dimension: 32 + 32 + 16 + 16 = 96
        concat_dim = 32 + 32 + 16 + 16

        # Concatenation MLP baseline
        self.mlp_fusion = nn.Sequential(
            nn.Linear(concat_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, out_action_dim)
        )

        # Optional Cross-Attention module
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.proj_tactile_32 = nn.Linear(16, 32)
        self.proj_lang_32 = nn.Linear(16, 32)
        self.cross_head = nn.Linear(32 * 4, out_action_dim)

    def forward(
        self,
        proprio: torch.Tensor,
        vision: torch.Tensor,
        tactile: torch.Tensor,
        language: torch.Tensor,
        missing_mask: Optional[Dict[str, bool]] = None,
        use_attention: bool = False
    ) -> torch.Tensor:
        """Forward pass with missing-modality mask zeroing."""
        # Check masks
        if missing_mask is not None:
            if missing_mask.get("vision", False):
                vision = torch.zeros_like(vision)
            if missing_mask.get("tactile", False):
                tactile = torch.zeros_like(tactile)
            if missing_mask.get("language", False):
                language = torch.zeros_like(language)

        e_proprio = self.enc_proprio(proprio)
        e_vis = self.enc_vision(vision)
        e_tac = self.enc_tactile(tactile)
        e_lang = self.enc_lang(language)

        if not use_attention:
            # Baseline concatenation
            fused = torch.cat([e_proprio, e_vis, e_tac, e_lang], dim=-1)
            return self.mlp_fusion(fused)
        else:
            # Cross attention between 4 modality tokens
            t_tac = self.proj_tactile_32(e_tac)
            t_lang = self.proj_lang_32(e_lang)
            tokens = torch.stack([e_proprio, e_vis, t_tac, t_lang], dim=1)  # (batch, 4, 32)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            return self.cross_head(attn_out.flatten(1))
