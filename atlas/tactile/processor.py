"""Tactile Intelligence Module for ATLAS HUMANOID AI 4D.
Extracts contact state, deformation features, and grasp confidence from GelSight and simulated contact sensors.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from atlas.deep_learning.models import CompactCNN


class TactileProcessor:
    def __init__(self, use_cnn: bool = True):
        self.use_cnn = use_cnn
        if use_cnn:
            self.model = CompactCNN(in_channels=3, out_classes=4)
            self.model.eval()

    def process_gelsight_frame(
        self,
        raw_tactile: np.ndarray,
        reference_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process a 64x64x3 GelSight tactile sensor image."""
        # 1. Difference from unpressed reference
        if reference_frame is not None:
            diff = np.abs(raw_tactile.astype(np.float32) - reference_frame.astype(np.float32))
        else:
            diff = raw_tactile.astype(np.float32)

        # 2. Total contact intensity
        intensity = float(np.mean(diff))
        contact_detected = (intensity > 5.0)

        # 3. Compact CNN feature extraction and state classification
        # State: 0: No Contact, 1: Initial Touch, 2: Stable Grasp, 3: Slip Warning
        if self.use_cnn:
            t_in = torch.tensor(raw_tactile, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                logits = self.model(t_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            predicted_state = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            predicted_state = 2 if contact_detected else 0
            confidence = min(1.0, intensity / 50.0)

        # Grasp stability confidence
        grasp_stability = 0.95 if (contact_detected and predicted_state in [1, 2]) else 0.1

        return {
            "contact_detected": contact_detected,
            "mean_intensity": intensity,
            "tactile_state": predicted_state,
            "confidence": confidence,
            "grasp_stability": grasp_stability
        }

    def process_simulated_contacts(
        self,
        hand_contacts: Dict[str, bool],
        grip_force: float = 0.0
    ) -> Dict[str, Any]:
        """Extract tactile proxy signals from MuJoCo physics contacts."""
        is_contact = hand_contacts.get("left_hand", False) or hand_contacts.get("right_hand", False)
        state = 2 if is_contact else 0
        return {
            "contact_detected": is_contact,
            "contact_force": grip_force,
            "tactile_state": state,
            "grasp_stability": 0.9 if is_contact else 0.0
        }
