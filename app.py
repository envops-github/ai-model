import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Log startup message
logging.info("Starting TinyLlama AI application...")

MODEL_ID = "TinyLlama/TinyLlama_v1.1"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    logging.info(f"✅ Successfully loaded model: {MODEL_ID}")
except Exception as e:
    logging.error(f"🔥 Failed to load model: {e}")
    sys.exit(1)

# Text generation function
def generate_text(prompt):
    logging.info(f"📝 Received prompt: {prompt}")
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"🤖 AI Response: {response}")
        return response
    except Exception as e:
        logging.error(f"❌ Error in text generation: {e}")
        return "Error processing your request."

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
