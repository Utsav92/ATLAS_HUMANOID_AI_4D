"""Unit tests for Milestone M4: NLP Instruction Parsing & Task Planner."""

import pytest
import numpy as np
from atlas.nlp.pipeline import NLPPipeline
from atlas.planning.task_planner import HumanoidTaskPlanner
from atlas.core.types import RobotState


def test_m4_nlp_canonical_command_parsing():
    nlp = NLPPipeline()
    cmd = "Walk to the red cube, pick it up, and place it on the shelf."
    plan = nlp.parse_instruction(cmd)

    assert plan.task == "PICK_AND_PLACE"
    assert plan.target_object == "red_cube"
    assert plan.destination == "shelf"
    assert plan.steps == [
        "LOCATE", "APPROACH", "STABILIZE", "REACH", "GRASP", "LIFT", "TRANSPORT", "PLACE"
    ]
    assert plan.status == "PENDING"


def test_m4_nlp_unsupported_rejection():
    nlp = NLPPipeline()
    cmd = "Fly to the moon and bake a cake."
    plan = nlp.parse_instruction(cmd)
    assert plan.task == "REJECTED"
    assert plan.status == "FAILED"


def test_m4_nlp_world_validation():
    nlp = NLPPipeline()
    known_world = ["blue_cube", "table", "shelf"]
    # Requesting red_cube which is not in known_world
    plan = nlp.parse_instruction("Pick up the red cube.", world_objects=known_world)
    assert plan.task == "REJECTED"


def test_m4_task_planner_fsm():
    nlp = NLPPipeline()
    plan = nlp.parse_instruction("Walk to the red cube, pick it up, and place it on the shelf.")
    planner = HumanoidTaskPlanner(plan)

    state = RobotState(base_position=np.array([0.0, 0.0, 0.79], dtype=np.float32))
    world_ctx = {"object_pose": [0.75, 0.05, 0.53]}

    # Check step 0: LOCATE
    assert planner.current_step_name == "LOCATE"
    ok, _ = planner.check_preconditions("LOCATE", state, world_ctx)
    assert ok is True

    # Advance
    next_step = planner.advance_step(postcondition_met=True)
    assert next_step == "APPROACH"

    # Advance through all steps
    for _ in range(len(plan.steps) - 1):
        planner.advance_step(postcondition_met=True)

    assert planner.current_step_name == "COMPLETED"
    assert plan.status == "COMPLETED"
