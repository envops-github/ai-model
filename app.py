import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import logging
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Log startup message
logging.info("Starting TinyLlama AI application...")

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

# Enable progress logging for model downloads
from transformers.utils.logging import set_verbosity_info
set_verbosity_info()

# Load model with 8-bit quantization
try:
    logging.info("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logging.info("⏳ Loading model with 8-bit quantization...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto"
    )

    logging.info(f"✅ Successfully loaded model: {MODEL_ID}")
except Exception as e:
    logging.error(f"🔥 Failed to load model: {e}")
    sys.exit(1)

# Function to handle timeouts
def timeout_wrapper(func, *args, timeout=30):
    """Executes a function with a timeout (30s)."""
    result = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            logging.error(f"❌ Timeout error: {e}")
            result[0] = "Error: Request timed out."

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        logging.warning("⚠️ Request exceeded timeout. Stopping execution.")
        return "Error: Request took too long."
    
    return result[0]

# AI Text generation function
def generate_text(prompt):
    logging.info(f"📝 Received prompt: {prompt}")

    def ai_response():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = timeout_wrapper(ai_response, timeout=30)
    
    if response.startswith("Error"):
        logging.error(response)
    else:
        logging.info(f"🤖 AI Response: {response}")
    
    return response

# UI for chatbot
ui = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your message...", lines=2, label="💬 Enter your prompt"),
    outputs=gr.Textbox(label="🤖 TinyLlama Response"),
    title="🌌 TinyLlama AI Chatbot",
    description="🔵 A lightweight AI chatbot powered by TinyLlama_v1.1.",
)

if __name__ == "__main__":
    logging.info("🚀 Starting UI on port 7860...")
    ui.launch(server_name="0.0.0.0", server_port=7860)
