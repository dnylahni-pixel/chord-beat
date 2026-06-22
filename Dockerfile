FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES="" \
    CHORDMINI_TIMEOUT="1800"

WORKDIR /app

# ۱. نصب پیش‌نیازهای سیستمی
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ۲. کلون و ستاپ ChordMini
RUN git clone https://github.com/ptnghia-j/ChordMini.git /app/ChordMini
WORKDIR /app/ChordMini
RUN git lfs pull
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ۳. ستاپ محیط اصلی Worker
WORKDIR /app
COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

# ---> تغییر طلایی: فورس کردن نصب نسخه CPU-Only برای هر دو پکیج <---
RUN pip install --no-cache-dir --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# ۴. انتقال فایل‌های هندلر و استاب
COPY cli_stub.py /app/cli_stub.py
COPY handler.py /app/handler.py

# ۵. پچ کردن اولیه مسيرهای ChordMini
RUN mkdir -p /app/ChordMini/src/utils && touch /app/ChordMini/src/utils/__init__.py
RUN cp /app/cli_stub.py /app/ChordMini/src/utils/cli.py
ENV PYTHONPATH="/app/ChordMini/src:/app/ChordMini"
RUN python -c "from utils.cli import bootstrap_cli, ensure_src_on_path; print('ChordMini CLI OK')"

# ۶. اطمینان از وجود فایل‌های چک‌پوینت هر دو مدل آکورد
RUN test -f /app/ChordMini/checkpoints/btc_model_best.pth && \
    test -f /app/ChordMini/checkpoints/2e1d_model_best.pth && \
    test -f /app/ChordMini/config/ChordMini.yaml && \
    test -f /app/ChordMini/src/evaluation/test.py

# ---> تغییر طلایی دوم: پری‌لود کردن مدل‌های beat_this در زمان بیلد <---
RUN python -c "from beat_this.inference import File2Beats; print('Preloading beat_this...'); [File2Beats(checkpoint_path=f'final{i}', device='cpu', dbn=False) for i in range(3)]; print('Preloading done.')"

CMD ["python", "-u", "/app/handler.py"]
