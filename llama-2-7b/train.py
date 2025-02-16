import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model

# ✅ Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Переменные модели
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./llama2-finetuned"
HF_REPO = "username/llama2-finetuned"  # НАДО ПОМЕНЯТЬ НА СВОЮ РЕПУ!

# ✅ Получение токена Hugging Face
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HF_TOKEN:
    logging.error("❌ HUGGINGFACE_TOKEN не установлен! Укажите токен в секретах GitHub.")
    exit(1)

# ✅ Авторизация в Hugging Face
login(HF_TOKEN)

# ✅ Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ Используем устройство: {device}")

# ✅ Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ Загрузка датасета (замени `dataset_name` на свой датасет)
dataset_name = "timdettmers/openassistant-guanaco"  # Или путь к локальному файлу .json/.csv
dataset = load_dataset(dataset_name, split="train")

# ✅ Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ✅ Настройка квантованной модели с LoRA (адаптивное дообучение)
logging.info("⏳ Загружаем модель с 4-битным квантованием...")

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)

# ✅ Настройка LoRA для экономии VRAM
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"],  # Дообучаем только ключевые параметры
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ✅ Параметры обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
    push_to_hub=True,  # Автоматическая загрузка в Hugging Face
    hub_model_id=HF_REPO,
    hub_token=HF_TOKEN
)

# ✅ Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# ✅ Запуск обучения
logging.info("🚀 Начинаем обучение модели...")
trainer.train()

# ✅ Сохранение и загрузка модели в Hugging Face Hub
logging.info(f"📤 Загружаем дообученную модель в Hugging Face: {HF_REPO}")
trainer.push_to_hub()
