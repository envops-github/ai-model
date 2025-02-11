FROM python:3.10

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision torchaudio \
    transformers bitsandbytes accelerate \
    gradio

COPY app.py /app/app.py

CMD ["python", "app.py"]
