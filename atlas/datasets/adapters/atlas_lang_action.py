"""Adapter for ATLAS Language-to-Action Dataset.
Generated from simulated instructions, task plans, object states, and robot trajectories.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.schema import DatasetManifest, LanguageActionSample


class AtlasLanguageActionAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="ATLAS Language-to-Action",
            source_id="atlas_lang_act_v1",
            version="1.0.0",
            license_spdx="Apache-2.0",
            robot_embodiment="Unitree G1 Multimodal",
            num_episodes=200,
            num_samples=2000,
            features=["instruction", "intent", "target_object", "destination", "subgoals", "initial_state", "target_state"],
            coordinate_frame="FLU",
            citation="ATLAS Embodied Language-to-Action Dataset (2026)"
        )

    def __init__(self, data_root: Optional[str] = None):
        super().__init__(data_root)
        self.samples = self._generate_corpus()

    def _generate_corpus(self) -> List[LanguageActionSample]:
        instructions = [
            ("Walk to the red cube, pick it up, and place it on the shelf.", "PICK_AND_PLACE", "red_cube", "shelf",
             ["LOCATE", "APPROACH", "STABILIZE", "REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE"]),
            ("Go to the blue block and grasp it.", "GRASP", "blue_block", "hand",
             ["LOCATE", "APPROACH", "STABILIZE", "REACH", "GRASP"]),
            ("Navigate to the table in front.", "NAVIGATE", "table", "table",
             ["LOCATE", "PLAN_PATH", "APPROACH", "STABILIZE"]),
            ("Stand still and maintain balance.", "STAND", "self", "origin",
             ["STABILIZE", "MONITOR_ZMP"]),
            ("Pick up the red cube from the table.", "PICK", "red_cube", "hand",
             ["LOCATE", "REACH", "GRASP", "LIFT"]),
            ("Transport the held object to the green container.", "TRANSPORT", "held_object", "green_container",
             ["PLAN_PATH", "APPROACH", "STABILIZE", "PLACE"]),
            ("Wave right hand at the user.", "GESTURE", "right_hand", "air",
             ["REACH", "OSCILLATE_WRIST", "RETRACT"]),
            ("Inspect the table surface with head camera.", "INSPECT", "table", "camera",
             ["PITCH_HEAD", "CAPTURE_FRAME", "ESTIMATE_POSES"])
        ]
        
        corpus = []
        for i in range(50):
            tmpl = instructions[i % len(instructions)]
            corpus.append(LanguageActionSample(
                sample_id=f"lang_act_{i:04d}",
                instruction=tmpl[0],
                intent=tmpl[1],
                target_object=tmpl[2],
                destination=tmpl[3],
                subgoals=list(tmpl[4]),
                initial_world_state={"robot_pos": [0.0, 0.0, 0.79], "object_pos": [0.75, 0.05, 0.53]},
                target_world_state={"object_pos": [0.85, 0.45, 0.55]},
                success=True
            ))
        return corpus

    def get_sample_count(self) -> int:
        return len(self.samples)

    def get_sample(self, index: int) -> LanguageActionSample:
        return self.samples[index % len(self.samples)]
