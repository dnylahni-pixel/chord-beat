import os
import uuid
import requests
import runpod
from beat_this.inference import File2Beats

f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)


def download_file(url: str) -> str:
    path = f"/tmp/{uuid.uuid4()}.mp3"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return path


def format_beats(beats, downbeats):
    result = []
    measure = 0
    beat_in_measure = 0
    downbeat_times = set(round(float(t), 3) for t in downbeats)

    for time_sec in beats:
        rounded_time = round(float(time_sec), 3)
        is_downbeat = rounded_time in downbeat_times

        if is_downbeat:
            measure += 1
            beat_in_measure = 1
        else:
            beat_in_measure += 1

        result.append({
            "beat": beat_in_measure,
            "time": round(float(time_sec), 2),
            "measure": measure,
            "isDownbeat": is_downbeat
        })

    return result


def run_chordmini(audio_path: str):
    return []


def handler(job):
    audio_path = None

    try:
        job_input = job.get("input", {})
        audio_url = job_input.get("audio_url")

        if not audio_url:
            return {
                "beats": [],
                "chords": []
            }

        audio_path = download_file(audio_url)
        beats_raw, downbeats_raw = f2b(audio_path)

        beats = format_beats(beats_raw, downbeats_raw)
        chords = run_chordmini(audio_path)

        return {
            "beats": beats,
            "chords": chords
        }

    except Exception:
        return {
            "beats": [],
            "chords": []
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
