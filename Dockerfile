# Use a minimal Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git g++ cmake make libstdc++6 libgcc-12 && \
    rm -rf /var/lib/apt/lists/*

# Ensure `GLIBCXX_3.4.32` is available
RUN ln -sf /usr/lib/gcc/x86_64-linux-gnu/12/libstdc++.so.6 /usr/lib/libstdc++.so.6

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

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py

# Expose port
EXPOSE 7860

# Run the application with unbuffered output
CMD ["python", "-u", "/app/app.py"]
