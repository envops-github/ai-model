FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git g++ libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU-only), Transformers, and Gradio
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers gradio accelerate bitsandbytes

WORKDIR /app

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "-u", "/app/app.py"]
