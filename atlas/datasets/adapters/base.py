"""Base Adapter class for ATLAS Dataset Hub.
Standardizes metadata, licensing, coordinate transformations, and embodiment validation.
"""

from __future__ import annotations
import abc
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from atlas.datasets.schema import DatasetManifest


class BaseDatasetAdapter(abc.ABC):
    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root
        self._manifest = self._build_manifest()

    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest

    @abc.abstractmethod
    def _build_manifest(self) -> DatasetManifest:
        pass

    @abc.abstractmethod
    def get_sample_count(self) -> int:
        pass

    @abc.abstractmethod
    def get_sample(self, index: int) -> Any:
        pass

    def validate_integrity(self) -> bool:
        """Verify that samples exist and conform to minimum schema."""
        count = self.get_sample_count()
        if count <= 0:
            return False
        sample = self.get_sample(0)
        return sample is not None

    def transform_coordinates(self, pos: np.ndarray, source_frame: str, target_frame: str = "FLU") -> np.ndarray:
        """Transform 3D position between coordinate conventions (e.g. RDF camera to FLU robot frame)."""
        if source_frame == target_frame:
            return pos.copy()
        
        # RDF (Right-Down-Forward, typical OpenCV camera) to FLU (Forward-Left-Up, MuJoCo/ROS)
        if source_frame == "RDF" and target_frame == "FLU":
            x_cam, y_cam, z_cam = pos
            return np.array([z_cam, -x_cam, -y_cam], dtype=np.float32)
        
        # FLU to RDF
        if source_frame == "FLU" and target_frame == "RDF":
            x, y, z = pos
            return np.array([-y, -z, x], dtype=np.float32)

        return pos.copy()

    def map_joints_to_g1(
        self,
        source_joint_names: List[str],
        source_joint_values: np.ndarray,
        target_g1_names: List[str]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Map joints from foreign embodiment into G1 joint space.
        Returns:
            (mapped_values, mapped_joint_names, missing_joint_names)
        """
        g1_values = np.zeros(len(target_g1_names), dtype=np.float32)
        name_to_val = {name: val for name, val in zip(source_joint_names, source_joint_values)}
        
        mapped = []
        missing = []

        for i, name in enumerate(target_g1_names):
            if name in name_to_val:
                g1_values[i] = name_to_val[name]
                mapped.append(name)
            else:
                missing.append(name)

        return g1_values, mapped, missing
