FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# نصب پیش‌نیازهای سیستمی مشترک
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ستاپ ChordMini و دانلود پکیج‌ها و چک‌پوینت‌ها از گیت ال‌اف‌اس
RUN git clone https://github.com/ptnghia-j/ChordMini.git /app/ChordMini
WORKDIR /app/ChordMini
RUN git lfs pull
RUN pip install --no-cache-dir -r requirements.txt

# ستاپ محیط اصلی برنامه و پکیج‌های مقصد
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# نصب اجباری نسخه CPU-Only فریم‌ورک Torch جهت بهینه‌سازی
RUN pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# اعمال زیرساخت ساب‌ماژول و استاب کامند لاین ChordMini
RUN mkdir -p /app/ChordMini/src/utils && touch /app/ChordMini/src/utils/__init__.py
COPY cli_stub.py /app/ChordMini/src/utils/cli.py

# انتقال هندلر یکپارچه
COPY handler.py /app/handler.py

# اعتبارسنجی نهایی صحت وجود فایل‌های کلیدی و چک‌پوینت‌ها در حین بیلد
RUN test -f /app/ChordMini/checkpoints/btc_model_best.pth && \
    test -f /app/ChordMini/checkpoints/2e1d_model_best.pth && \
    test -f /app/ChordMini/config/ChordMini.yaml && \
    test -f /app/ChordMini/src/evaluation/test.py

ENV PYTHONPATH="/app/ChordMini/src:/app/ChordMini"

CMD ["python", "-u", "/app/handler.py"]
