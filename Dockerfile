# Use a minimal Python base image
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU-only), Transformers, and Gradio
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers gradio

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "-u", "app.py"]
