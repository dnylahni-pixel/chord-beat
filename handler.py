import runpod, requests, subprocess, os
from beat_this.inference import File2Beats

f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)

def parse_lab(path):
    chords = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                chords.append({
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "chord": parts[2]
                })
    return chords

def handler(job):
    url = job["input"]["audio_url"]
    audio_path = "/tmp/song.mp3"
    with open(audio_path, "wb") as f:
        f.write(requests.get(url).content)

    beats, downbeats = f2b(audio_path)

    out_dir = "/tmp/chord_out"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "python", "ChordMini/src/evaluation/test.py",
        "--model_type", "ChordNet",
        "--checkpoint", "ChordMini/checkpoints/2e1d_model_best.pth",
        "--config", "ChordMini/config/ChordMini.yaml",
        "--audio_dir", audio_path,
        "--save_dir", out_dir,
        "--use_overlap", "--use_gaussian",
        "--kernel_size", "9",
        "--vote_aggregation", "logit",
        "--min_segment_duration", "0.5",
        "--smooth_predictions"
    ], check=True)

    lab_files = [f for f in os.listdir(out_dir) if f.endswith(".lab")]
    chords = parse_lab(os.path.join(out_dir, lab_files[0])) if lab_files else []

    return {
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "chords": chords
    }

runpod.serverless.start({"handler": handler})
