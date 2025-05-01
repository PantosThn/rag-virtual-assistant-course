# frontend/ui.py — Enhanced Gradio UI for RAG chatbot focused on Greek economy
import gradio as gr
import requests
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

DESCRIPTION = """
<div style="text-align: center">
    <h1>📊 Greek Economy RAG Assistant</h1>
    <p style="font-size: 18px; color: gray">
        Explore inflation, GDP, debt, unemployment, and more through intelligent retrieval-augmented dialogue.
    </p>
</div>
"""

# --- Backend call (supports streaming simulation) ---------------------------
def chat_with_rag(message, history):
    try:
        resp = requests.post(BACKEND_URL, json={"question": message})
        resp.raise_for_status()
        answer = resp.json().get("answer", "No answer returned.")
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history


# Streaming simulation (not token-by-token unless backend supports it)
def stream_with_rag(message, history):
    yield "", history + [{"role": "user", "content": message}]
    try:
        resp = requests.post(BACKEND_URL, json={"question": message})
        resp.raise_for_status()
        answer = resp.json().get("answer", "No answer returned.")
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    streamed = ""
    for word in answer.split():
        streamed += word + " "
        time.sleep(0.03)  # simulate streaming effect
        yield "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": streamed.strip()}
        ]


# Reset function
def reset():
    return "", []


# --- Interface --------------------------------------------------------------
with gr.Blocks(title="Greek Economy Chatbot", theme=gr.themes.Soft()) as demo:
    gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="💬 Dialogue", type="messages", height=400)
            user_input = gr.Textbox(placeholder="Ask something about Greek economic trends…", show_label=False)

            gr.Examples(
                examples=[
                    "What are the implications of Greece's request to exempt €500 million in defense spending from EU fiscal rules?",
                    "How significant is Moody's upgrade of Greece's economy to investment grade?",
                    "What are the key takeaways from the IMF's 2025 Article IV consultation with Greece?",
                    "How is Greece planning to repay its first bailout loans ahead of schedule?"
                ],
                inputs=user_input,
            )

            with gr.Row():
                submit = gr.Button("📤 Send", variant="primary")
                clear = gr.Button("🔄 Reset")


    submit.click(stream_with_rag, [user_input, chatbot], [user_input, chatbot], show_progress=True)
    user_input.submit(stream_with_rag, [user_input, chatbot], [user_input, chatbot], show_progress=True)
    clear.click(reset, outputs=[user_input, chatbot])

if __name__ == "__main__":
    demo.launch()