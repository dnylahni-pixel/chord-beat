FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    CHORDMINI_TIMEOUT="1800" \
    XDG_CACHE_HOME=/app/cache \
    TORCH_HOME=/app/cache/torch \
    HF_HOME=/app/cache/hf \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    build-essential \
    gcc \
    g++ \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN git clone https://github.com/ptnghia-j/ChordMini.git /app/ChordMini

WORKDIR /app/ChordMini
RUN git lfs pull
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY . .

RUN mkdir -p /app/ChordMini/src/utils && \
    touch /app/ChordMini/src/utils/__init__.py && \
    cp /app/cli_stub.py /app/ChordMini/src/utils/cli.py

ENV PYTHONPATH="/app/ChordMini/src:/app/ChordMini"

RUN mkdir -p /app/cache /tmp/matplotlib

RUN python -c "from utils.cli import bootstrap_cli, ensure_src_on_path; print('ChordMini CLI OK')"

RUN test -f /app/ChordMini/checkpoints/2e1d_model_best.pth && \
    test -f /app/ChordMini/config/ChordMini.yaml && \
    test -f /app/ChordMini/src/evaluation/test.py

CMD ["python", "-u", "handler.py"]
