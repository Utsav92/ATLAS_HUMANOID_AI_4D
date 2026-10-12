"""Unit tests for Milestone M9: Integrated Humanoid Task Execution."""

import pytest
import os
import json
from pathlib import Path
from atlas.planning.end_to_end_executor import EndToEndExecutor
from atlas.core.config import CONFIG


def test_m9_full_17_step_execution():
    executor = EndToEndExecutor()
    cmd = "Walk to the red cube, pick it up, and place it on the shelf."
    log = executor.run_demonstration(cmd)

    assert log.task_plan is not None
    assert log.task_plan.task == "PICK_AND_PLACE"
    assert len(log.states) > 20
    assert len(log.actions) == len(log.states)
    assert len(log.explanations) >= 16
    assert log.success is True

    # Verify TouchDesigner replay file created
    replay_file = Path(CONFIG.td.ipc_dir) / "episode_replay.json"
    assert replay_file.exists()
    assert replay_file.stat().st_size > 0

    with open(replay_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["success"] is True
        assert len(data["frames"]) > 0
        assert "explanations" in data
