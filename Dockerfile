FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir setuptools==80.9.0 && pip install --no-cache-dir -r ChordMini/requirements.txt
CMD ["python", "-u", "handler.py"]
