FROM python:3.10-alpine as builder

RUN apk add --no-cache git wget curl

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1+cpu transformers gradio

FROM python:3.10-alpine

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH="/root/.local/bin:$PATH"

COPY app.py /app/app.py

EXPOSE 7860

CMD ["python", "app.py"]
