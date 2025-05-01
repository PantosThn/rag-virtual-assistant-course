# frontend/ui.py — Chat-style Gradio frontend for your FastAPI RAG backend
import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

def chat_with_rag(message, history):
    try:
        resp = requests.post(BACKEND_URL, json={"question": message})
        resp.raise_for_status()
        answer = resp.json().get("answer", "No answer returned.")
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
    return "", history + [[message, answer]]

with gr.Blocks(title="🧠 RAG Chat") as demo:
    gr.Markdown("## 💬 Ask your RAG Assistant")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question and press Enter…")
    msg.submit(chat_with_rag, [msg, chatbot], [msg, chatbot])
