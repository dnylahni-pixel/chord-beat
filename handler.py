import os
import uuid
import requests
import runpod
from beat_this.inference import File2Beats

MODEL_CONFIGS = [
    {"model_id": "final0", "checkpoint_path": "final0"},
    {"model_id": "final1", "checkpoint_path": "final1"},
    {"model_id": "final2", "checkpoint_path": "final2"},
]

MODELS = [
    {
        "model_id": config["model_id"],
        "checkpoint_path": config["checkpoint_path"],
        "runner": File2Beats(
            checkpoint_path=config["checkpoint_path"],
            device="cpu",
            dbn=False,
        ),
    }
    for config in MODEL_CONFIGS
]


def download_file(url: str) -> str:
    path = f"/tmp/{uuid.uuid4()}.mp3"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return path


def serialize_events(events):
    serialized = []

    for event in events:
        serialized.append(float(event))

    return serialized


def run_chordmini(audio_path: str):
    return []


def empty_response():
    return {
        "models": [
            {
                "model_id": config["model_id"],
                "checkpoint_path": config["checkpoint_path"],
                "dbn": False,
                "beats": [],
                "downbeats": [],
            }
            for config in MODEL_CONFIGS
        ],
        "chords": [],
    }


def handler(job):
    audio_path = None

    try:
        job_input = job.get("input", {})
        audio_url = job_input.get("audio_url")

        if not audio_url:
            return empty_response()

        audio_path = download_file(audio_url)

        models_output = []
        for model in MODELS:
            beats_raw, downbeats_raw = model["runner"](audio_path)

            models_output.append(
                {
                    "model_id": model["model_id"],
                    "checkpoint_path": model["checkpoint_path"],
                    "dbn": False,
                    "beats": serialize_events(beats_raw),
                    "downbeats": serialize_events(downbeats_raw),
                }
            )

        chords = run_chordmini(audio_path)

        return {
            "models": models_output,
            "chords": chords,
        }

    except Exception:
        return empty_response()

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
