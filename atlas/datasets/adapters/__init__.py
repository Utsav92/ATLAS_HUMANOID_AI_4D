"""Dataset adapters for all 13 framework and benchmark sources + ATLAS Language-to-Action."""

from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.adapters.humanoid_motion import (
    G1MovesAdapter,
    G1ManipulationAdapter,
    MuJoCoTrajectoriesAdapter
)
from atlas.datasets.adapters.slam_datasets import (
    TUMRGBDAdapter,
    EuRoCMAVAdapter,
    KITTIAdapter,
    TartanAirAdapter
)
from atlas.datasets.adapters.tactile_datasets import (
    YCBSightAdapter,
    SparshAdapter
)
from atlas.datasets.adapters.manipulation_datasets import (
    DROIDAdapter,
    OpenXEmbodimentAdapter,
    LeRobotAdapter,
    MinariAdapter
)
from atlas.datasets.adapters.atlas_lang_action import (
    AtlasLanguageActionAdapter
)

__all__ = [
    "BaseDatasetAdapter",
    "G1MovesAdapter",
    "G1ManipulationAdapter",
    "MuJoCoTrajectoriesAdapter",
    "TUMRGBDAdapter",
    "EuRoCMAVAdapter",
    "KITTIAdapter",
    "TartanAirAdapter",
    "YCBSightAdapter",
    "SparshAdapter",
    "DROIDAdapter",
    "OpenXEmbodimentAdapter",
    "LeRobotAdapter",
    "MinariAdapter",
    "AtlasLanguageActionAdapter"
]
