import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Log startup message
logging.info("Starting TinyLlama AI application...")

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

# Enable progress logging for model downloads
from transformers.utils.logging import set_verbosity_info
set_verbosity_info()

# Force BitsAndBytes to use CPU
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "cpu_max_split_size_mb:512"

# Check if bitsandbytes is available
try:
    import bitsandbytes as bnb
    logging.info("✅ bitsandbytes is installed.")
except ImportError:
    logging.error("❌ bitsandbytes is missing! Install it using `pip install -U bitsandbytes`.")
    sys.exit(1)

# Force CPU usage
device = torch.device("cpu")

try:
    logging.info("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logging.info("⏳ Loading model with 8-bit quantization on CPU...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": device}
    )

    logging.info(f"✅ Successfully loaded model on {device}: {MODEL_ID}")
except Exception as e:
    logging.error(f"🔥 Failed to load model: {e}")
    sys.exit(1)
