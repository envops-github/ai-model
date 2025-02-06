FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch transformers gradio

WORKDIR /app

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
