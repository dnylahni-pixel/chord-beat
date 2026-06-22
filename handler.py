import os
import sys
import uuid
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

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

APP_ROOT = Path(__file__).resolve().parent
CHORDMINI_ROOT = APP_ROOT / "ChordMini"

CONFIG_PATH = CHORDMINI_ROOT / "config" / "ChordMini.yaml"
CHECKPOINT_PATH = CHORDMINI_ROOT / "checkpoints" / "2e1d_model_best.pth"
TEST_SCRIPT = CHORDMINI_ROOT / "src" / "evaluation" / "test.py"

MODEL_TYPE = "ChordNet"
TIMEOUT_SECONDS = int(os.environ.get("CHORDMINI_TIMEOUT", "1800"))


def download_file(url: str, dst_dir: Path) -> Path:
    parsed = urlparse(url)
    name = Path(parsed.path).name

    if not name or "." not in name:
        name = f"{uuid.uuid4().hex}.mp3"

    path = dst_dir / name

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if path.stat().st_size < 1024:
        raise RuntimeError(f"Downloaded file is too small: {path.stat().st_size} bytes")

    return path


def serialize_events(events):
    serialized = []

    for event in events:
        try:
            serialized.append(float(event))
        except TypeError:
            serialized.append([float(value) for value in event])

    return serialized


def parse_lab_file(lab_path: Path):
    chords = []

    with open(lab_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=2)

            if len(parts) < 3:
                continue

            start_str, end_str, chord = parts

            chords.append(
                {
                    "start": float(start_str),
                    "end": float(end_str),
                    "chord": chord,
                }
            )

    return chords


def patch_chordmini_test_script():
    if not TEST_SCRIPT.exists():
        raise RuntimeError(f"ChordMini test.py not found: {TEST_SCRIPT}")

    content = TEST_SCRIPT.read_text()

    if "current_dir in sys.path" in content:
        return

    lines = content.splitlines()
    insert_idx = 0

    for index, line in enumerate(lines):
        if line.strip().startswith("from __future__"):
            insert_idx = index + 1

    injection_lines = [
        "import sys",
        "import os",
        "current_dir = os.path.dirname(os.path.abspath(__file__))",
        "if current_dir in sys.path: sys.path.remove(current_dir)",
        "sys.path.insert(0, '/app/ChordMini/src')",
        "sys.path.insert(1, '/app/ChordMini')",
    ]

    patched = lines[:insert_idx] + injection_lines + lines[insert_idx:]
    TEST_SCRIPT.write_text("\n".join(patched) + "\n")


def run_chordmini(audio_path: Path, save_dir: Path):
    if not CHORDMINI_ROOT.exists():
        raise RuntimeError(f"ChordMini directory not found: {CHORDMINI_ROOT}")

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"ChordMini config not found: {CONFIG_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"ChordMini checkpoint not found: {CHECKPOINT_PATH}")

    patch_chordmini_test_script()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = f"/app/ChordMini/src:/app/ChordMini:{env.get('PYTHONPATH', '')}"

    if "CHORDMINI_MODEL_TYPE" in env:
        del env["CHORDMINI_MODEL_TYPE"]

    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--audio_dir", str(audio_path.parent),
        "--save_dir", str(save_dir),
        "--config", str(CONFIG_PATH),
        "--checkpoint", str(CHECKPOINT_PATH),
        "--model_type", MODEL_TYPE,
        "--use_overlap",
        "--use_gaussian",
        "--kernel_size", "9",
        "--vote_aggregation", "logit",
        "--min_segment_duration", "0.5",
        "--smooth_predictions",
    ]

    started = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(CHORDMINI_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=TIMEOUT_SECONDS,
    )

    elapsed = time.time() - started

    if result.returncode != 0:
        raise RuntimeError(
            f"ChordMini failed with returncode {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    lab_files = sorted(save_dir.glob("*.lab"))

    if not lab_files:
        raise RuntimeError(
            "ChordMini completed but no .lab file was generated\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    return parse_lab_file(lab_files[0]), elapsed


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
    work_dir = None

    try:
        job_input = job.get("input", {})
        audio_url = job_input.get("audio_url") or job_input.get("url")

        if not audio_url:
            return empty_response()

        work_dir = Path(tempfile.gettempdir()) / f"beatthis_chordmini_{uuid.uuid4().hex}"
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = download_file(audio_url, input_dir)

        models_output = []

        for model in MODELS:
            beats_raw, downbeats_raw = model["runner"](str(audio_path))

            models_output.append(
                {
                    "model_id": model["model_id"],
                    "checkpoint_path": model["checkpoint_path"],
                    "dbn": False,
                    "beats": serialize_events(beats_raw),
                    "downbeats": serialize_events(downbeats_raw),
                }
            )

        chords = []
        debug = {}

        try:
            chords, chordmini_time = run_chordmini(audio_path, output_dir)
            debug["chordmini_time_sec"] = round(chordmini_time, 2)
        except Exception as chord_error:
            debug["chordmini_error"] = str(chord_error)

        response = {
            "models": models_output,
            "chords": chords,
        }

        if debug:
            response["debug"] = debug

        return response

    except Exception as error:
        response = empty_response()
        response["error"] = str(error)
        return response

    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
