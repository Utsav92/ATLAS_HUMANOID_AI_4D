"""Adapters for Tactile Sensing & Representation Learning:
8. YCB-Sight
9. Sparsh
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.schema import DatasetManifest, TactileSample


class YCBSightAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="YCB-Sight",
            source_id="ycb_sight_tactile",
            version="1.0",
            license_spdx="MIT",
            robot_embodiment="GelSight Tactile Sensor",
            num_episodes=50,
            num_samples=3500,
            features=["tactile_image_raw", "contact_mask", "normal_field", "shear_field"],
            coordinate_frame="SensorSurface",
            citation="YCB-Sight: High-Resolution Tactile Dataset for YCB Objects (2022)",
            url="https://github.com/Robo-Touch/YCB-Sight",
            github_url="https://github.com/Robo-Touch/YCB-Sight",
            purpose="Tactile sensing and object understanding"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> TactileSample:
        contact = (index % 3 != 0)
        return TactileSample(
            sample_id=f"ycb_sight_{index:04d}",
            timestamp=float(index) * 0.05,
            sensor_type="GelSight",
            taxel_shape=(64, 64, 3),
            contact_force=3.2 if contact else 0.0,
            slip_detected=(index % 7 == 0),
            contact_state=1 if contact else 0
        )


class SparshAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="Sparsh",
            source_id="sparsh_tactile_ssl",
            version="1.0",
            license_spdx="Apache-2.0",
            robot_embodiment="Multi-Sensor Tactile Benchmark",
            num_episodes=80,
            num_samples=6000,
            features=["touch_embedding", "pressure_distribution", "slip_indicator", "contact_geometry"],
            coordinate_frame="SensorSurface",
            citation="Sparsh: Towards Self-Supervised Touch Representations (2024)",
            url="https://github.com/facebookresearch/sparsh",
            github_url="https://github.com/facebookresearch/sparsh",
            hf_url="https://huggingface.co/facebook/sparsh",
            purpose="Tactile representation learning"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> TactileSample:
        return TactileSample(
            sample_id=f"sparsh_repr_{index:04d}",
            timestamp=float(index) * 0.05,
            sensor_type="Sparsh",
            taxel_shape=(32, 32),
            contact_force=2.1,
            slip_detected=False,
            contact_state=1
        )
