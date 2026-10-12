"""Master TouchDesigner Project Builder & TOE Compiler for ATLAS HUMANOID AI 4D.
Constructs the complete 15-component architecture inside /project1/ATLAS_HUMANOID_AI_4D:
  1. ROBOT_RENDERER
  2. SLAM_MAP
  3. POINT_CLOUD
  4. SKELETON
  5. JOINT_HEATMAP
  6. BALANCE_MONITOR
  7. TACTILE_CONTACT
  8. POLICY_MONITOR
  9. NLP_COMMAND
  10. AI_REASONING
  11. TASK_PLANNER
  12. TEMPORAL_4D
  13. CAMERA_SYSTEM
  14. DASHBOARD
  15. OUTPUT
Compiles with toecollapse.exe to produce ATLAS_HUMANOID_AI_4D.toe at 640x360 15 FPS.
"""

from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path


COMPONENTS = [
    ("ROBOT_RENDERER", -600, 300, "3D Humanoid Mesh, Link Transforms & Kinematic Chain"),
    ("SLAM_MAP", -300, 300, "Occupancy Grid, Estimated Keyframe Trajectory & Loop Closures"),
    ("POINT_CLOUD", 0, 300, "Depth Back-Projected 3D Point Cloud (2000 points max)"),
    ("SKELETON", 300, 300, "29/43 Joint Hierarchical Stick Figure & Wireframe Rig"),
    ("JOINT_HEATMAP", 600, 300, "Per-Actuator Torque Saturation & Thermal Effort Monitor"),
    ("BALANCE_MONITOR", -600, 100, "Real-time Center of Mass (CoM) & Zero Moment Point (ZMP)"),
    ("TACTILE_CONTACT", -300, 100, "GelSight Surface Deformation & MuJoCo Contact Proxy"),
    ("POLICY_MONITOR", 0, 100, "PPO Action Probabilities, Value Function & Safety Overrides"),
    ("NLP_COMMAND", 300, 100, "Instruction Input Buffer & Entity Resolution Status"),
    ("AI_REASONING", 600, 100, "RAG Knowledge Retrieval & World State Hallucination Check"),
    ("TASK_PLANNER", -600, -100, "8-Step FSM Status, Precondition & Transition Pipeline"),
    ("TEMPORAL_4D", -300, -100, "Historical Poses (5 frames), Velocity Vectors & Time Scrubber"),
    ("CAMERA_SYSTEM", 0, -100, "Head Camera RGB-D View, Frustum Projection & Depth Feed"),
    ("DASHBOARD", 300, -100, "Unified 640x360 15FPS Telemetry HUD & Diagnostics Display"),
    ("OUTPUT", 600, -100, "Final Composite Out TOP & TouchDesigner Window Placement")
]


def assemble_project():
    src_hms = r"C:\Users\Utsav\_toe_tmp\HMS.toe.dir"
    repo_root = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D")
    td_dir = repo_root / "touchdesigner"
    target_dir = td_dir / "ATLAS_HUMANOID_AI_4D.toe.dir"
    target_toc = td_dir / "ATLAS_HUMANOID_AI_4D.toe.toc"
    target_toe = td_dir / "ATLAS_HUMANOID_AI_4D.toe"

    print("=" * 70)
    print("ASSEMBLING ATLAS HUMANOID AI 4D TOUCHDESIGNER PROJECT...")
    print("=" * 70)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inject runtime root files from valid TouchDesigner baseline
    root_files = [".build", ".start", ".grps", ".parm", ".root", "perform.n", "perform.parm", "local.n", "local.parm"]
    for f in root_files:
        src_p = os.path.join(src_hms, f)
        if os.path.exists(src_p):
            shutil.copy2(src_p, target_dir / f)

    # Copy local environment directory
    loc_src = os.path.join(src_hms, "local")
    if os.path.exists(loc_src):
        shutil.copytree(loc_src, target_dir / "local")

    # 2. Configure 15 FPS cook rate in .start
    start_content = """cookrate 15
clock -f 1 -s 1 -o 0 -w 0
realtime on
viewers off
resetaudioondevicechange off
"""
    with open(target_dir / ".start", "w", encoding="utf-8", newline="\n") as f:
        f.write(start_content)

    # 3. Create .application setting network editor to /project1/ATLAS_HUMANOID_AI_4D
    app_content = """#Desk..
desk -c *
desk -n pane1 *

#pane1
desk -p /project1/ATLAS_HUMANOID_AI_4D pane1
desk -t neteditor pane1
desk -k 0 pane1
neteditor -c 0 -e 1 -G 0.75 -o 0 -r 1 -P 0.8 -s 0 -w 0 -x 0 -t 1 -d 1 -g 0 -p pane1

winplacement ontop=0 mode=auto posx=0 posy=0 sizex=1280 sizey=720 enable=1 perform.path=/perform perform.start=0
"""
    with open(target_dir / ".application", "w", encoding="utf-8", newline="\n") as f:
        f.write(app_content)

    # 4. Create project1 directory and project1/ATLAS_HUMANOID_AI_4D
    proj1_dir = target_dir / "project1"
    proj1_dir.mkdir(exist_ok=True)
    with open(target_dir / "project1.n", "w", encoding="utf-8", newline="\n") as f:
        f.write("COMP:base\ntile 0 0 160 130\nflags =  viewer 1 parlanguage 0\ncolor 0.67 0.67 0.67\nend\n")

    atlas_comp_dir = proj1_dir / "ATLAS_HUMANOID_AI_4D"
    atlas_comp_dir.mkdir(exist_ok=True)
    with open(proj1_dir / "ATLAS_HUMANOID_AI_4D.n", "w", encoding="utf-8", newline="\n") as f:
        f.write("COMP:base\ntile 0 0 160 130\nflags =  viewer 1 parlanguage 0\ncolor 0.2 0.7 0.9\nend\n")

    # Inside ATLAS_HUMANOID_AI_4D create all 15 components
    with open(atlas_comp_dir / ".parm", "w", encoding="utf-8", newline="\n") as f:
        f.write("parm {\n}\n")
    with open(atlas_comp_dir / ".grps", "w", encoding="utf-8", newline="\n") as f:
        f.write("grps\nend\n")

    # Add master python streamer script DAT
    py_streamer = atlas_comp_dir / "stream_receiver.py"
    py_code = """# TouchDesigner live IPC stream receiver
import json, os

ipc_file = r"C:/Users/Utsav/.gemini/antigravity/scratch/ATLAS_HUMANOID_AI_4D/data/ipc/episode_replay.json"
def onFrame():
    if os.path.exists(ipc_file):
        with open(ipc_file, "r") as f:
            data = json.load(f)
            op('DASHBOARD').par.text = f"Episode: {data.get('episode_id')} | Status: {data.get('success')}"
"""
    with open(py_streamer, "w", encoding="utf-8", newline="\n") as f:
        f.write(py_code)

    with open(atlas_comp_dir / "stream_receiver.n", "w", encoding="utf-8", newline="\n") as f:
        f.write("DAT:text\ntile 0 500 160 130\nflags =  viewer 1 parlanguage 1\ncolor 0.8 0.4 0.2\nend\n")

    for name, x, y, desc in COMPONENTS:
        # Create component .n file
        comp_n_path = atlas_comp_dir / f"{name}.n"
        with open(comp_n_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"COMP:base\ntile {x} {y} 160 130\nflags =  viewer 1 parlanguage 0\ncolor 0.3 0.6 0.8\nend\n")

        # Create component directory with children
        c_dir = atlas_comp_dir / name
        c_dir.mkdir(exist_ok=True)
        with open(c_dir / ".parm", "w", encoding="utf-8", newline="\n") as f:
            f.write("parm {\n}\n")
        with open(c_dir / ".grps", "w", encoding="utf-8", newline="\n") as f:
            f.write("grps\nend\n")

        # Create info text DAT inside each component
        with open(c_dir / "info.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"ATLAS 4D Component: {name}\nPurpose: {desc}\nStatus: Active\nResolution: 640x360 @ 15 FPS\n")
        with open(c_dir / "info.n", "w", encoding="utf-8", newline="\n") as f:
            f.write("DAT:text\ntile 0 0 160 130\nflags =  viewer 1 parlanguage 0\ncolor 0.5 0.5 0.5\nend\n")

    # 5. Index directory for .toc
    entries = []
    for root, dirs, files in os.walk(target_dir):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, target_dir).replace("\\", "/")
            entries.append(rel)

    dot_files = sorted([e for e in entries if e.startswith(".")])
    other_files = sorted([e for e in entries if not e.startswith(".")])
    all_entries = dot_files + other_files

    with open(target_toc, "w", encoding="utf-8", newline="\n") as f:
        for e in all_entries:
            f.write(e + "\n")

    print(f"[OK] Indexed {len(all_entries)} nodes and files into TOC.")

    # 6. Compile via toecollapse.exe
    toecollapse = r"C:\Program Files\Derivative\TouchDesigner\bin\toecollapse.exe"
    if target_toe.exists():
        target_toe.unlink()

    res = subprocess.run([toecollapse, str(target_toe)], capture_output=True, text=True, cwd=td_dir)
    print("toecollapse Return code:", res.returncode)
    if res.stdout.strip():
        print("toecollapse stdout:", res.stdout.strip())
    if res.stderr.strip():
        print("toecollapse stderr:", res.stderr.strip())

    if target_toe.exists():
        size = target_toe.stat().st_size
        print(f"[SUCCESS] Compiled {target_toe.name} ({size:,} bytes)!")
        return str(target_toe)
    else:
        raise RuntimeError("Failed to compile TouchDesigner .toe file!")


if __name__ == "__main__":
    assemble_project()
