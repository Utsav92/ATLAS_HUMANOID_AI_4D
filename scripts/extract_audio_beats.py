import subprocess
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

def main():
    video_path = "videos/real_human_salsa_31s.mp4"
    wav_path = "data/salsa_audio_30s.wav"
    os.makedirs("data", exist_ok=True)

    # 1. Extract 16kHz mono audio using ffmpeg
    print("Extracting audio from video...")
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-ss", "00:00:00", "-t", "00:00:30",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path, "-y"
    ]
    subprocess.run(cmd, check=True)

    # 2. Load wav
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32) / 32768.0

    # 3. Audio energy envelope & onset detection
    hop_len = int(sr * 0.01) # 10ms window (100 Hz)
    num_frames = len(audio) // hop_len
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        chunk = audio[i * hop_len : (i + 1) * hop_len]
        energy[i] = np.sum(chunk ** 2)

    # Envelope derivative for onsets
    energy_diff = np.diff(energy, prepend=energy[0])
    onset_env = np.maximum(0, energy_diff)
    onset_env = onset_env / (np.max(onset_env) + 1e-6)

    # Find peaks corresponding to musical salsa beats (~180-200 BPM -> ~0.30 - 0.35s between beats)
    min_distance = int(0.25 * (sr / hop_len)) # at least 250ms apart
    peaks, props = find_peaks(onset_env, height=0.15, distance=min_distance)
    beat_times = peaks * (hop_len / sr)

    print(f"Detected {len(beat_times)} salsa musical beats across 30 seconds.")
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        avg_interval = np.median(intervals)
        estimated_bpm = 60.0 / avg_interval
        print(f"Estimated Salsa Tempo: {estimated_bpm:.1f} BPM (Average beat interval: {avg_interval:.3f}s)")

    # Save beat data
    np.savez("data/salsa_beat_grid.npz", beat_times=beat_times, onset_env=onset_env, hop_sr=(sr/hop_len))
    print("Saved audio beat grid to data/salsa_beat_grid.npz")

if __name__ == "__main__":
    main()
