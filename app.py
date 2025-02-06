import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import logging
import time
import traceback

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

# Define text generation function
def generate_text(prompt):
    logging.info(f"📝 Received prompt: {prompt}")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"🤖 AI Response: {response}")
        return response
    except Exception as e:
        logging.error(f"❌ Error in text generation: {e}")
        return "Error processing your request."

# Gradio UI
ui = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your message...", lines=2, label="💬 Enter your prompt"),
    outputs=gr.Textbox(label="🤖 TinyLlama Response"),
    title="🌌 TinyLlama AI Chatbot",
    description="🔵 A lightweight AI chatbot powered by TinyLlama_v1.1.",
)

# Start Gradio UI and prevent the container from exiting
try:
    if __name__ == "__main__":
        logging.info("🚀 Starting UI on port 7860...")
        ui.launch(server_name="0.0.0.0", server_port=7860)

        # Keep the process alive even if UI crashes
        while True:
            time.sleep(60)
except Exception as e:
    logging.error(f"❌ Unhandled exception: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)
