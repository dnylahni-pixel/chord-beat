import os
import sys
import uuid
import time
import shutil
import subprocess
import traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod
from beat_this.inference import File2Beats

# =============================================================================
# GLOBALS & PATHS
# =============================================================================
APP_ROOT = Path(__file__).resolve().parent
CHORDMINI_ROOT = APP_ROOT / "ChordMini"

CONFIG_PATH = CHORDMINI_ROOT / "config" / "ChordMini.yaml"
TEST_SCRIPT = CHORDMINI_ROOT / "src" / "evaluation" / "test.py"
TIMEOUT_SECONDS = int(os.environ.get("CHORDMINI_TIMEOUT", "1800"))

# تنظیمات مدل‌های بیت (beat_this)
MODEL_CONFIGS = [
    {"model_id": "final0", "checkpoint_path": "final0"},
    {"model_id": "final1", "checkpoint_path": "final1"},
    {"model_id": "final2", "checkpoint_path": "final2"},
]

# پیش‌بارگذاری ۳ مدل بیت در مموری
print("[INIT] Loading beat_this models...")
BEAT_MODELS = [
    {
        "model_id": config["model_id"],
        "checkpoint_path": config["checkpoint_path"],
        "runner": File2Beats(checkpoint_path=config["checkpoint_path"], device="cpu", dbn=False),
    }
    for config in MODEL_CONFIGS
]
print("[INIT] All beat_this models loaded.")


# =============================================================================
# UTILITIES
# =============================================================================
def download_file(url: str, dst_dir: Path) -> Path:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name or "." not in name:
        name = f"audio_{uuid.uuid4().hex[:8]}.mp3"
    dst = dst_dir / name
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst

def serialize_events(events):
    return [float(event) for event in events]

def parse_lab_file(lab_path: Path):
    chords = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=2)
            if len(parts) < 3: continue
            try:
                chords.append({
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "chord": parts[2]
                })
            except ValueError:
                pass
    return chords

# =============================================================================
# CHORDMINI ENGINE
# =============================================================================
def patch_test_py():
    """تزریق پچ مسیر به test.py برای جلوگیری از تداخل مسیرهای محلی"""
    if TEST_SCRIPT.exists():
        original_content = TEST_SCRIPT.read_text()
        if "current_dir in sys.path" not in original_content:
            lines = original_content.splitlines()
            insert_idx = next((i + 1 for i, line in enumerate(lines) if line.strip().startswith("from __future__")), 0)
            injection_lines = [
                "import sys, os",
                "current_dir = os.path.dirname(os.path.abspath(__file__))",
                "if current_dir in sys.path: sys.path.remove(current_dir)",
                "sys.path.insert(0, '/app/ChordMini/src')",
                "sys.path.insert(1, '/app/ChordMini')"
            ]
            new_lines = lines[:insert_idx] + injection_lines + lines[insert_idx:]
            TEST_SCRIPT.write_text("\n".join(new_lines) + "\n")

# اجرای یک بار پچ در زمان استارت شدن کانتینر
patch_test_py()

def run_chordmini_model(audio_path: Path, output_dir: Path, model_type: str, checkpoint_name: str, use_smooth_logits: bool):
    """اجرای داینامیک یک مدل از ChordMini (مدیریت پارامترهای متفاوت هر مدل)"""
    checkpoint_path = CHORDMINI_ROOT / "checkpoints" / checkpoint_name
    
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = f"/app/ChordMini/src:/app/ChordMini:{env.get('PYTHONPATH', '')}"
    if "CHORDMINI_MODEL_TYPE" in env:
        del env["CHORDMINI_MODEL_TYPE"]

    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--audio_dir", str(audio_path.parent),
        "--save_dir", str(output_dir),
        "--config", str(CONFIG_PATH),
        "--checkpoint", str(checkpoint_path),
        "--model_type", model_type,
        "--use_overlap",
        "--use_gaussian",
        "--kernel_size", "9",
        "--vote_aggregation", "logit",
        "--min_segment_duration", "0.5",
        "--smooth_predictions",
    ]
    
    # فقط مدل BTC به این فلگ نیاز دارد
    if use_smooth_logits:
        cmd.append("--smooth_logits")

    result = subprocess.run(
        cmd, cwd=str(CHORDMINI_ROOT), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT_SECONDS
    )

    if result.returncode != 0:
        raise RuntimeError(f"ChordMini ({model_type}) failed. STDERR: {result.stderr[:500]}")

    lab_files = sorted(output_dir.glob("*.lab"))
    if not lab_files:
        raise RuntimeError(f"ChordMini ({model_type}) generated no .lab files.")

    return parse_lab_file(lab_files[0])

# =============================================================================
# MAIN HANDLER
# =============================================================================
def empty_response():
    return {
        "models": [
            {
                "model_id": conf["model_id"],
                "checkpoint_path": conf["checkpoint_path"],
                "dbn": False,
                "beats": [],
                "downbeats": []
            } for conf in MODEL_CONFIGS
        ],
        "chords": [
            {"model_id": "BTC", "data": []},
            {"model_id": "ChordNet", "data": []}
        ]
    }

def handler(job):
    job_input = job.get("input", {})
    audio_url = job_input.get("audio_url") or job_input.get("url")

    if not audio_url:
        return empty_response()

    work_dir = Path("/tmp") / f"merged_worker_{uuid.uuid4().hex}"
    input_dir = work_dir / "input"
    output_btc = work_dir / "output_btc"
    output_chordnet = work_dir / "output_chordnet"
    
    for d in [input_dir, output_btc, output_chordnet]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        # ۱. دانلود فایل
        audio_path = download_file(audio_url, input_dir)

        # ۲. پردازش ۳ مدل Beat
        models_output = []
        for model in BEAT_MODELS:
            beats_raw, downbeats_raw = model["runner"](str(audio_path))
            models_output.append({
                "model_id": model["model_id"],
                "checkpoint_path": model["checkpoint_path"],
                "dbn": False,
                "beats": serialize_events(beats_raw),
                "downbeats": serialize_events(downbeats_raw),
            })

        # ۳. پردازش مدل BTC آکورد
        btc_chords = run_chordmini_model(
            audio_path=audio_path,
            output_dir=output_btc,
            model_type="BTC",
            checkpoint_name="btc_model_best.pth",
            use_smooth_logits=True
        )

        # ۴. پردازش مدل ChordNet آکورد
        chordnet_chords = run_chordmini_model(
            audio_path=audio_path,
            output_dir=output_chordnet,
            model_type="ChordNet",
            checkpoint_name="2e1d_model_best.pth",
            use_smooth_logits=False
        )

        # ۵. بازگرداندن خروجی نهایی یکپارچه
        return {
            "models": models_output,
            "chords": [
                {"model_id": "BTC", "data": btc_chords},
                {"model_id": "ChordNet", "data": chordnet_chords}
            ]
        }

    except Exception as e:
        print(f"[ERROR] Job Failed: {e}")
        print(traceback.format_exc())
        return empty_response()

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
