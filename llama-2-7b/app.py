import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import logging
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

# ✅ Проверяем доступность CUDA
if torch.cuda.is_available():
    device = "cuda"
    logging.info("✅ Using CUDA for inference!")
else:
    logging.error("❌ No CUDA available! Please check GPU drivers.")
    exit(1)

# ✅ Освобождаем VRAM перед загрузкой
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ✅ Загрузка токенизатора
try:
    logging.info("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    
    # ✅ Фикс отсутствующего PAD-токена
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("✅ Assigned eos_token as pad_token to tokenizer.")
except Exception as e:
    logging.error(f"🔥 Failed to load tokenizer: {e}")
    exit(1)

# ✅ Загрузка модели с 4-битным квантованием
try:
    logging.info("⏳ Loading model with 4-bit quantization on GPU...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # ⚡ 4-битное квантование (экономия VRAM)
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,  # ⚡ Экономия VRAM
        low_cpu_mem_usage=True,  # ⚡ Экономия памяти при загрузке
        token=HF_TOKEN
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

        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,  # ✅ Теперь работает!
            max_length=512  # ✅ Фикс максимальной длины
        ).to(device)

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
    title="🌌 EnvOps AI Chatbot ",
    description="🚀 A chatbot powered by Llama 2 (7B) with 4-bit quantization on GPU by EnvOps.",
)

# ✅ Запуск Gradio UI
try:
    if __name__ == "__main__":
        logging.info("🚀 Starting UI on port 7860...")
        ui.launch(server_name="0.0.0.0", server_port=7860)
except Exception as e:
    logging.error(f"❌ Unhandled exception: {e}")
    logging.error(traceback.format_exc())
    exit(1)
