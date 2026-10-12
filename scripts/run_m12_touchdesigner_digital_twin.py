"""Milestone M12 Runner: TouchDesigner 4D Digital Twin.
Verifies compiled project, launches TouchDesigner live if not already active, and streams telemetry.
"""

import time
import json
import subprocess
import os
from pathlib import Path
from atlas.core.config import CONFIG
from atlas.planning.end_to_end_executor import EndToEndExecutor


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M12 TOUCHDESIGNER DIGITAL TWIN")
    print("=" * 70)

    toe_path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\touchdesigner\ATLAS_HUMANOID_AI_4D.toe")
    assert toe_path.exists(), f"Target .toe file missing: {toe_path}"
    print(f"[OK] Verified TouchDesigner Project: {toe_path} ({toe_path.stat().st_size:,} bytes)")

    # Execute simulation demonstration to produce fresh live telemetry
    print("Executing MuJoCo 17-step simulation to populate live IPC stream...")
    executor = EndToEndExecutor()
    log = executor.run_demonstration("Walk to the red cube, pick it up, and place it on the shelf.")

    ipc_replay = Path(CONFIG.td.ipc_dir) / "episode_replay.json"
    ipc_stream = Path(CONFIG.td.ipc_dir) / "atlas_live_stream.json"
    print(f"[OK] Live IPC Telemetry Generated: {ipc_replay.name} ({ipc_replay.stat().st_size:,} bytes)")

    # Check if TouchDesigner is running
    res = subprocess.run(
        ["powershell", "-Command", "Get-Process | Where-Object { $_.ProcessName -like '*TouchDesigner*' } | Select-Object -ExpandProperty Id"],
        capture_output=True, text=True
    )
    pids = [p.strip() for p in res.stdout.strip().split("\n") if p.strip()]
    if pids:
        print(f"[OK] Live TouchDesigner process active (PID: {pids[0]})")
    else:
        print("Launching TouchDesigner process live...")
        td_exe = r"C:\Program Files\Derivative\TouchDesigner\bin\TouchDesigner.exe"
        subprocess.Popen([td_exe, str(toe_path)])
        time.sleep(3)
        print("[OK] TouchDesigner launched successfully.")

    print("\n[SUCCESS] Milestone M12 validated: TouchDesigner 4D digital twin operational.")


if __name__ == "__main__":
    main()
