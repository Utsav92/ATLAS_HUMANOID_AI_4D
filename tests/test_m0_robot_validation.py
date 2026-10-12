"""Unit tests for Milestone M0: Windows setup and MuJoCo robot validation."""

import pytest
import numpy as np
from atlas.simulation.world import AtlasWorld
from atlas.core.types import RobotState


def test_m0_world_initialization():
    world = AtlasWorld()
    assert world.model is not None
    assert world.data is not None
    assert world.robot is not None
    assert world.robot.model.nu >= 29


def test_m0_standing_equilibrium():
    world = AtlasWorld()
    state = world.reset()
    assert isinstance(state, RobotState)
    assert not world.robot.is_fallen()
    assert state.base_position[2] > 0.6  # Standing height > 0.6m

    # Step for 50 control steps (1.0 second simulation) with nominal stand control
    for _ in range(50):
        state = world.step(world.robot.stand_ctrl)

    assert not world.robot.is_fallen()
    assert state.base_position[2] > 0.6


def test_m0_vision_rendering():
    world = AtlasWorld(render_width=160, render_height=120)
    rgb = world.render_rgb()
    depth = world.render_depth()
    assert rgb.shape == (120, 160, 3)
    assert depth.shape == (120, 160)
    assert rgb.dtype == np.uint8
    assert depth.dtype == np.float32


def test_m0_object_poses():
    world = AtlasWorld()
    cube_pos = world.get_object_pose("red_cube")
    table_pos = world.get_object_pose("table")
    assert cube_pos is not None
    assert table_pos is not None
    assert cube_pos.shape == (3,)
    assert cube_pos[2] > table_pos[2]  # Cube is above the table
