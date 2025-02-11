FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y python3 python3-pip git && \
    apt install -y wget

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir transformers bitsandbytes accelerate gradio

COPY app.py /app/app.py

ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

CMD ["python3", "app.py"]
