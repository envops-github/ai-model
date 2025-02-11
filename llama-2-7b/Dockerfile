FROM python:3.10

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision torchaudio \
    transformers bitsandbytes accelerate \
    gradio

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY app.py /app/app.py

ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

CMD ["python", "app.py"]
