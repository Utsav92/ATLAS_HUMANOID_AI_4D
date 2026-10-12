# TouchDesigner live IPC stream receiver
import json, os

ipc_file = r"C:/Users/Utsav/.gemini/antigravity/scratch/ATLAS_HUMANOID_AI_4D/data/ipc/episode_replay.json"
def onFrame():
    if os.path.exists(ipc_file):
        with open(ipc_file, "r") as f:
            data = json.load(f)
            op('DASHBOARD').par.text = f"Episode: {data.get('episode_id')} | Status: {data.get('success')}"
