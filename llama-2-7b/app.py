import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import logging
import time
import traceback
from huggingface_hub import login

# ✅ Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("🚀 Starting Llama 2 AI application...")

# ✅ Переменные модели
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

# ✅ Получение токена Hugging Face из переменной окружения
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    logging.error("❌ HUGGINGFACE_TOKEN is not set! Please provide a valid Hugging Face API token.")
    exit(1)

# ✅ Авторизация в Hugging Face
logging.info("🔑 Logging into Hugging Face...")
login(HF_TOKEN)

# ✅ Принудительное использование CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

# ✅ Загрузка токенизатора
try:
    logging.info("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
except Exception as e:
    logging.error(f"🔥 Failed to load tokenizer: {e}")
    exit(1)

# ✅ Загрузка модели с 4-битной квантованием
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
        device_map={"": device},
        use_auth_token=HF_TOKEN  # Передаем токен для доступа к закрытой модели
    )

    logging.info(f"✅ Successfully loaded model on {device}: {MODEL_ID}")
except Exception as e:
    logging.error(f"🔥 Failed to load model: {e}")
    exit(1)

# ✅ Функция генерации текста
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

# ✅ Gradio UI
ui = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your message...", lines=2, label="💬 Enter your prompt"),
    outputs=gr.Textbox(label="🤖 Llama 2 Response"),
    title="🌌 Llama 2 AI Chatbot",
    description="🚀 A chatbot powered by Llama 2 (7B) with 4-bit quantization.",
)

# ✅ Запуск Gradio UI и поддержка активности контейнера
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
