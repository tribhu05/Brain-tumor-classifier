# Brain Tumor Classifier -- inference/training container
FROM python:3.11-slim

WORKDIR /app

# System dependencies required by Pillow/matplotlib/OpenCV-adjacent libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/config.yaml"]
