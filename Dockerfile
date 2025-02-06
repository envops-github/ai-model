FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git g++ gcc cmake make libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "transformers>=4.45.1"

# Install PyTorch (CPU-only), Transformers, Gradio
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir gradio accelerate intel_extension_for_pytorch

# Clone and install BitsAndBytes from source (CPU mode)
RUN git clone --depth 1 -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
    cd bitsandbytes && \
    pip install -r requirements-dev.txt && \
    cmake -DCOMPUTE_BACKEND=cpu -S . && \
    make && \
    pip install -e .

WORKDIR /app

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "-u", "/app/app.py"]
