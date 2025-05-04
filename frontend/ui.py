# frontend/ui.py — Greek‑Economy Chatbot with live status bubbles
import os, time, requests, gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")

DESCRIPTION = """
<div style="text-align:center">
  <h1>📈 Greek Economy RAG Assistant</h1>
  <p style="font-size:18px;color:gray">Explore insights on GDP, inflation, debt, NRRP, HAPS, OCW, and more.</p>
</div>
"""


def query_backend(q: str):
    r = requests.post(BACKEND_URL, json={"question": q}, timeout=45)
    r.raise_for_status()
    return r.json().get("answer", "No answer returned.")


def stream_with_rag(message: str, history: list):
    message = (message or "").strip()
    if not message:
        yield "", history
        return

    # 1) Echo user
    history = history + [{"role": "user", "content": message}]
    yield "", history

    # 2) Add temporary assistant state
    history = history + [{"role": "assistant", "content": "⏳ Thinking…"}]
    yield "", history

    # 3) Call backend
    try:
        answer = query_backend(message)
    except Exception as e:
        answer = f"Error: {e}"

    # 4) Stream word‑by‑word
    streamed = ""
    for word in answer.split():
        streamed += word + " "
        history[-1]["content"] = streamed.strip()
        time.sleep(0.02)
        yield "", history

    # final yield to ensure last word rendered
    yield "", history


def reset():
    return "", []

with gr.Blocks(title="Greek Economy Chatbot", theme=gr.themes.Soft()) as demo:
    gr.HTML(DESCRIPTION)

    chatbot    = gr.Chatbot(height=420, label="Dialogue", type="messages")
    user_input = gr.Textbox(placeholder="Ask about Greek economic trends…", show_label=False)

    gr.Examples([
    "what does the imf say about greek recovery?",
    "What was the average headline inflation rate for Greece in 2024, and how did it compare to the euro area average?",
    "According to the 2024 Ageing Report, how is Greece’s public pension expenditure as a percentage of GDP expected to evolve by 2070, and how does it compare to the EU average?"
    ], inputs=user_input)

    with gr.Row():
        send_btn  = gr.Button("📤 Send", variant="primary")
        clear_btn = gr.Button("🔄 Reset")

    send_btn.click(stream_with_rag, [user_input, chatbot], [user_input, chatbot])
    user_input.submit(stream_with_rag, [user_input, chatbot], [user_input, chatbot])
    clear_btn.click(reset, outputs=[user_input, chatbot])

if __name__ == "__main__":
    demo.launch()