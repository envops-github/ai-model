import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import logging
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("🚀 Starting Llama 2 AI application...")

# Model selection
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# Load tokenizer
logging.info("⏳ Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load model with 4-bit quantization
try:
    logging.info("⏳ Loading model with 4-bit quantization on CPU...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": device}
    )

    logging.info(f"✅ Successfully loaded model on {device}: {MODEL_ID}")
except Exception as e:
    logging.error(f"🔥 Failed to load model: {e}")
    exit(1)

# Define text generation function
def generate_text(prompt):
    logging.info(f"📝 Received prompt: {prompt}")

    try:
        system_instruction = "You are an AI assistant. Answer questions concisely and informatively.\n"
        full_prompt = system_instruction + prompt

        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.9
        )
        
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
    outputs=gr.Textbox(label="🤖 Llama 2 Response"),
    title="🌌 Llama 2 AI Chatbot",
    description="🚀 A chatbot powered by Llama 2 (7B) with 4-bit quantization.",
)

# Start Gradio UI and keep the app running
try:
    if __name__ == "__main__":
        logging.info("🚀 Starting UI on port 7860...")
        ui.launch(server_name="0.0.0.0", server_port=7860)

        while True:
            time.sleep(60)
except Exception as e:
    logging.error(f"❌ Unhandled exception: {e}")
    logging.error(traceback.format_exc())
    exit(1)
