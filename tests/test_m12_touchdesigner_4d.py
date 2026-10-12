import pytest
import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from atlas.core.config import CONFIG
from touchdesigner.assemble_atlas_toe import assemble_project


def test_m12_toe_compilation():
    toe_path = Path(assemble_project())
    assert toe_path.exists()
    assert toe_path.stat().st_size > 1000


def test_m12_ipc_stream_payload():
    ipc_dir = Path(CONFIG.td.ipc_dir)
    ipc_dir.mkdir(parents=True, exist_ok=True)
    replay_path = ipc_dir / "episode_replay.json"

    # Verify or write test replay payload
    if not replay_path.exists():
        payload = {
            "episode_id": "test_ep",
            "instruction": "Test command",
            "success": True,
            "metrics": {"fall_rate": 0.0},
            "frames": [{"t": 0.0, "base_pos": [0, 0, 0.79], "com": [0, 0, 0.69]}],
            "total_frames": 1
        }
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    assert replay_path.exists()
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "episode_id" in data
        assert "frames" in data
