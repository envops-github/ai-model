import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load TinyLlama model and tokenizer
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Text generation function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Custom CSS for dark blue-gray theme
custom_css = """
body {
    background-color: #1a1d2e;
    color: white;
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    background: #1a1d2e;
    border-radius: 10px;
    padding: 20px;
}

textarea {
    background: #252a41 !important;
    color: white !important;
    border: 1px solid #3b405f !important;
}

button {
    background: #3b82f6 !important;
    color: white !important;
    border-radius: 5px !important;
    font-size: 16px !important;
}

button:hover {
    background: #2563eb !important;
}
"""

# Gradio UI
ui = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(placeholder="Type your message...", lines=2, label="💬 Enter your prompt"),
    outputs=gr.Textbox(label="🤖 TinyLlama Response"),
    title="🌌 TinyLlama AI Chatbot by EnvOps", 
    description="🔵 A lightweight AI chatbot powered by TinyLlama (1.1B).",
    theme="soft",
    css=custom_css,  # Apply custom styling
)

# Run the UI
if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
