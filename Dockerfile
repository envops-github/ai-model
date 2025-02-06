FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && apt-get install -y libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers gradio accelerate bitsandbytes-cpu

WORKDIR /app

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "-u", "/app/app.py"]
