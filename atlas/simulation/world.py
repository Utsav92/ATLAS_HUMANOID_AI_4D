"""MuJoCo Simulation World for ATLAS HUMANOID AI 4D.
Manages the simulation environment, rendering (RGB + Depth), camera projection, and physics stepping.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import mujoco

from atlas.core.types import RobotState, MultimodalObservation
from atlas.simulation.robot import G1Robot
from atlas.core.config import CONFIG


class AtlasWorld:
    def __init__(self, xml_path: Optional[str] = None, render_width: int = 320, render_height: int = 240):
        if xml_path is None:
            xml_path = str(CONFIG.robot.task_scene_xml_path)
            if not os.path.exists(xml_path):
                xml_path = str(CONFIG.robot.model_xml_path)
        
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot = G1Robot(self.model, self.data)

        self.render_width = render_width
        self.render_height = render_height
        self._renderer: Optional[mujoco.Renderer] = None

        # Reset to nominal standing pose
        self.reset()

    def get_renderer(self) -> Optional[mujoco.Renderer]:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
            except Exception:
                self._renderer = None
        return self._renderer

    def reset(self) -> RobotState:
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset_to_stand()
        return self.robot.get_state()

    def step(self, ctrl: Optional[np.ndarray] = None, n_substeps: int = 10) -> RobotState:
        """Advance the physics by n_substeps (e.g. 10 * 0.002s = 0.02s / 50Hz)."""
        if ctrl is not None:
            clipped_ctrl = self.robot.clip_ctrl(ctrl)
            self.data.ctrl[:len(clipped_ctrl)] = clipped_ctrl

        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.robot.get_state()

    def render_rgb(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render RGB image from specified camera or default camera view."""
        try:
            renderer = self.get_renderer()
            if renderer is not None:
                if camera_name is not None:
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                    renderer.update_scene(self.data, camera=cam_id)
                else:
                    renderer.update_scene(self.data)
                return renderer.render()
        except Exception:
            pass
        return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

    def render_depth(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render depth map (metric depth in meters)."""
        try:
            renderer = self.get_renderer()
            if renderer is not None:
                renderer.enable_depth_rendering()
                if camera_name is not None:
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                    renderer.update_scene(self.data, camera=cam_id)
                else:
                    renderer.update_scene(self.data)
                depth = renderer.render()
                renderer.disable_depth_rendering()
                return depth
        except Exception:
            pass
        return np.zeros((self.render_height, self.render_width), dtype=np.float32)

    def get_object_pose(self, name: str) -> Optional[np.ndarray]:
        """Query ground truth 3D position [x, y, z] of an object body in the simulation."""
        try:
            body_id = self.model.body(name).id
            return np.array(self.data.xpos[body_id], dtype=np.float32)
        except KeyError:
            return None

    def get_multimodal_observation(self, render_vision: bool = True) -> MultimodalObservation:
        """Package a complete synchronized observation of the robot and world."""
        state = self.robot.get_state()
        rgb = None
        depth = None
        missing = []

        if render_vision:
            try:
                rgb = self.render_rgb()
                depth = self.render_depth()
            except Exception as e:
                missing.extend(["rgb_image", "depth_map"])
        else:
            missing.extend(["rgb_image", "depth_map"])

        # Ground truth object detections for reference
        detected_objects = []
        for obj_name in ["red_cube", "table", "shelf"]:
            pos = self.get_object_pose(obj_name)
            if pos is not None:
                detected_objects.append({
                    "name": obj_name,
                    "position": pos.tolist(),
                    "confidence": 1.0
                })

        return MultimodalObservation(
            timestamp=float(self.data.time),
            robot_state=state,
            rgb_image=rgb,
            depth_map=depth,
            estimated_pose=np.array([state.base_position[0], state.base_position[1], 0.0], dtype=np.float32),
            detected_objects=detected_objects,
            missing_modalities=missing
        )
