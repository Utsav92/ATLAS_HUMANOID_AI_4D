"""Unit tests for Milestone M8: Tactile Intelligence and Multimodal Fusion."""

import pytest
import numpy as np
import torch

from atlas.tactile.processor import TactileProcessor
from atlas.fusion.multimodal_fusion import MultimodalFusionNet


def test_m8_tactile_processing():
    processor = TactileProcessor(use_cnn=True)

    # Synthetic GelSight image (64x64x3)
    raw_gelsight = np.random.randint(10, 40, (64, 64, 3), dtype=np.uint8)
    res = processor.process_gelsight_frame(raw_gelsight)

    assert "contact_detected" in res
    assert "grasp_stability" in res
    assert isinstance(res["confidence"], float)

    # Test simulated MuJoCo contacts
    sim_res = processor.process_simulated_contacts({"left_hand": False, "right_hand": True}, grip_force=4.5)
    assert sim_res["contact_detected"] is True
    assert sim_res["grasp_stability"] > 0.5


def test_m8_multimodal_fusion_forward():
    net = MultimodalFusionNet(
        proprio_dim=50, vision_dim=32, tactile_dim=16, language_dim=16, out_action_dim=43
    )

    batch_size = 4
    proprio = torch.randn(batch_size, 50)
    vision = torch.randn(batch_size, 32)
    tactile = torch.randn(batch_size, 16)
    language = torch.randn(batch_size, 16)

    # 1. Baseline concatenation MLP
    out_mlp = net(proprio, vision, tactile, language, use_attention=False)
    assert out_mlp.shape == (batch_size, 43)

    # 2. Cross attention
    out_attn = net(proprio, vision, tactile, language, use_attention=True)
    assert out_attn.shape == (batch_size, 43)


def test_m8_missing_modality_mask():
    net = MultimodalFusionNet(
        proprio_dim=50, vision_dim=32, tactile_dim=16, language_dim=16, out_action_dim=43
    )
    proprio = torch.randn(2, 50)
    vision = torch.randn(2, 32)
    tactile = torch.randn(2, 16)
    language = torch.randn(2, 16)

    # Mask vision and tactile dropped
    mask = {"vision": True, "tactile": True}
    out = net(proprio, vision, tactile, language, missing_mask=mask)
    assert out.shape == (2, 43)
    assert not torch.isnan(out).any()
