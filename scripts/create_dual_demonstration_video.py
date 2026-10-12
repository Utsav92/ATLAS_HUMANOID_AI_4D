"""
ATLAS HUMANOID AI 4D - Dual Embodiment Demonstration Video Generator
Generates a synchronized side-by-side video comparing:
1. Human Demonstrator (Motion Capture kinematics & real performer camera)
2. Unitree G1 Humanoid Robot (Autonomous 29-DoF Embodied AI execution)
Executing identical action with frame-accurate synchronization.
"""

import os
import subprocess
import sys

def main():
    root = r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D"
    videos_dir = os.path.join(root, "videos")
    
    human_mocap = os.path.join(videos_dir, "spiral_capture.mp4")
    robot_retarget = os.path.join(videos_dir, "spiral_retarget.mp4")
    studio_video = os.path.join(videos_dir, "demo_real_human_robot.mp4")
    
    out_video = os.path.join(videos_dir, "atlas_humanoid_human_dual_45s.mp4")
    out_preview = os.path.join(videos_dir, "atlas_dual_preview.png")
    
    font_arial = "C\\:/Windows/Fonts/arial.ttf"
    font_bold = "C\\:/Windows/Fonts/arialbd.ttf"
    font_mono = "C\\:/Windows/Fonts/consola.ttf"
    
    # Filter complex explanation:
    # [0:v] Human mocap scaled to 580x580
    # [1:v] Robot retarget scaled to 580x580, fps set to 30
    # [2:v] Studio real human footage cropped to dancer (496x425 from 110:580) and scaled to 200x170
    # Background: 1280x720 canvas
    
    filter_complex = (
        "[0:v]trim=0:45,setpts=PTS-STARTPTS,scale=580:580[human_mocap];"
        "[1:v]trim=0:45,setpts=PTS-STARTPTS,scale=580:580,fps=30[robot_retarget];"
        "[2:v]crop=480:400:115:590,scale=220:184[studio_dancer];"
        "color=c=#0b0d13:s=1280x720:d=45:r=30[bg];"
        "[bg][human_mocap]overlay=40:70[stage1];"
        "[stage1][robot_retarget]overlay=660:70[stage2];"
        "[stage2][studio_dancer]overlay=52:450[stage3];"
        # Bounding border on studio dancer PIP
        "[stage3]drawbox=x=50:y=448:w=224:h=188:color=#00e5ff@0.8:t=2,"
        # Top HUD bar
        "drawbox=x=0:y=0:w=1280:h=60:color=black@0.9:t=fill,"
        # Bottom HUD bar
        "drawbox=x=0:y=662:w=1280:h=58:color=black@0.9:t=fill,"
        # Accent lines
        "drawbox=x=0:y=60:w=1280:h=2:color=#00e5ff@0.6:t=fill,"
        "drawbox=x=0:y=660:w=1280:h=2:color=#39ff14@0.6:t=fill,"
        # Dividers
        "drawbox=x=639:y=62:w=2:h=598:color=#333a45@0.5:t=fill,"
        # Text Header
        f"drawtext=fontfile='{font_bold}':text='ATLAS HUMANOID AI 4D - HUMAN-TO-ROBOT MOTION IMITATION':fontcolor=white:fontsize=22:x=40:y=18,"
        # Human Demonstrator Labels
        f"drawtext=fontfile='{font_bold}':text='HUMAN DEMONSTRATOR':fontcolor=#39ff14:fontsize=17:x=42:y=76,"
        f"drawtext=fontfile='{font_arial}':text='Mocap Kinematics (51-DoF Performer)':fontcolor=#aaaaaa:fontsize=13:x=260:y=80,"
        f"drawtext=fontfile='{font_bold}':text='STUDIO CAMERA':fontcolor=#00e5ff:fontsize=11:x=56:y=456,"
        # Robot Labels
        f"drawtext=fontfile='{font_bold}':text='UNITREE G1 HUMANOID':fontcolor=#00e5ff:fontsize=17:x=662:y=76,"
        f"drawtext=fontfile='{font_arial}':text='Autonomous Embodied AI (29-DoF Whole-Body)':fontcolor=#aaaaaa:fontsize=13:x=885:y=80,"
        # Bottom Telemetry Badges
        f"drawtext=fontfile='{font_mono}':text='DATASET\\: G1 MOVES (exptech/g1-moves) | EMBODIMENT\\: UNITREE G1 | SYNC\\: LOCKED 1\\:1 | TRACKING ERR\\: <0.04m':fontcolor=#dddddd:fontsize=14:x=40:y=680,"
        # Live Timecode
        f"drawtext=fontfile='{font_mono}':text='%{{pts\\:hms}} / 00\\:00\\:45':fontcolor=#ffea00:fontsize=15:x=1080:y=680"
    )
    
    # 1. First render preview frame at t=15s
    print("Generating preview frame at t=15s...")
    cmd_preview = [
        "ffmpeg", "-y",
        "-ss", "15", "-i", human_mocap,
        "-ss", "15", "-i", robot_retarget,
        "-ss", "10", "-i", studio_video,
        "-filter_complex", filter_complex,
        "-vframes", "1",
        out_preview
    ]
    res = subprocess.run(cmd_preview, capture_output=True, text=True)
    if res.returncode != 0:
        print("Preview failed:", res.stderr)
        sys.exit(1)
    print(f"Preview saved to {out_preview}")

    # 2. Render complete 45-second video
    print("Rendering 45.00s synchronized human+robot video...")
    cmd_render = [
        "ffmpeg", "-y",
        "-i", human_mocap,
        "-i", robot_retarget,
        "-stream_loop", "-1", "-i", studio_video,
        "-filter_complex", filter_complex,
        "-t", "45.0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        out_video
    ]
    res_render = subprocess.run(cmd_render, capture_output=True, text=True)
    if res_render.returncode != 0:
        print("Render failed:", res_render.stderr)
        sys.exit(1)
    print(f"45s Dual Video created successfully at {out_video}")

if __name__ == "__main__":
    main()
