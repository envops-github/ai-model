FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git g++ gcc cmake make libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Ensure `GLIBCXX_3.4.32` is available
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/libstdc++.so.6

# Install the correct version of PyTorch (CPU-only) and IPEX 2.5
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.5.0 torchvision==0.16.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir intel_extension_for_pytorch==2.5.0 transformers==4.45.1 gradio accelerate 

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
