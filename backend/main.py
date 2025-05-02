# backend/main.py – GPT‑4‑o RAG with multi‑query + grading + fallback web search
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, requests
from pathlib import Path
from functools import lru_cache
from typing import List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
import gradio as gr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings   # fallback for Groq


# 1. ENV + model pick

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not (OPENAI_API_KEY or GROQ_API_KEY):
    raise RuntimeError("Provide OPENAI_API_KEY or GROQ_API_KEY in .env")

@lru_cache
def pick_models():
    if OPENAI_API_KEY:
        llm_full = ChatOpenAI(model_name="gpt-4o",
                              temperature=0,
                              openai_api_key=OPENAI_API_KEY)
        llm_mini = ChatOpenAI(model_name="gpt-4o-mini",
                              temperature=0,
                              openai_api_key=OPENAI_API_KEY)
        emb      = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        src = "GPT‑4‑o (full & mini)"
    else:
        llm_full = llm_mini = ChatOpenAI(
            model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
            temperature=0,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1")
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        src = "Groq + HF‑embeddings (fallback)"
    print(f"[backend] ✅ Using {src}")
    return emb, llm_full, llm_mini

embeddings, llm_full, llm_mini = pick_models()


# 2. Vector‑store (Chroma) + ingestion

DATA_DIR   = ROOT / "data"; DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR = ROOT / "chroma_store"

vectorstore = Chroma(persist_directory=str(CHROMA_DIR),
                     embedding_function=embeddings)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1200, chunk_overlap=0
)

def ingest(docs: List[Document]) -> int:
    if not docs:
        return 0
    chunks = [c for c in splitter.split_documents(docs)
              if c.page_content.strip()]
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return len(chunks)

# first‑time ingest of PDFs/TXTs dropped into /data
docs: List[Document] = []
for fp in DATA_DIR.rglob("*"):
    if fp.suffix.lower() == ".pdf":
        docs.extend(PyPDFLoader(str(fp)).load())
    elif fp.suffix.lower() in {".txt", ".text"}:
        docs.extend(TextLoader(str(fp)).load())
if docs:
    print(f"[backend] +{ingest(docs):,} chunks ingested from /data")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("[backend] 🏁 Chroma ready  (docs:", vectorstore._collection.count(), ")")


# 3. Prompt templates & chains

MULTI_REWRITE = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user query in three *different* ways that could improve "
     "vector search recall. One per line, no bullets."),
    ("human", "Original: {question}\nRewrites:")
]) | llm_full | StrOutputParser()

GRADER = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY 'Yes' if the document might help answer the question, else 'No'."),
    ("human", "Document:\n{document}\n\nQuestion: {question}")
]) | llm_mini | StrOutputParser()

GEN_PROMPT = ChatPromptTemplate.from_template(
    "Answer using only the context below. "
    "If insufficient, share any partial info you have **and** explicitly say "
    "\"I don't know\" where details are missing.\n\n{context}\n\n"
    "Question: {question}\nAnswer:"
) | llm_full | StrOutputParser()

web_search = TavilySearchResults(k=3, tavily_api_key=TAVILY_API_KEY) \
             if TAVILY_API_KEY else None


# 4. LangGraph workflow

class State(TypedDict):
    question:   str
    rewrites:   List[str]
    documents:  List[Document]
    generation: str
    route:      str          # control flow

# ---  nodes ---------------------------------------------------------------
def _rewrite(s: State):
    rew = MULTI_REWRITE.invoke({"question": s["question"]})
    lines = [r.strip() for r in rew.splitlines() if r.strip()]
    return {"rewrites": [s["question"], *lines]}

def _retrieve(s: State):
    all_docs: List[Document] = []
    for q in s["rewrites"]:
        all_docs.extend(retriever.invoke(q))
    return {"documents": all_docs}

def _grade(s: State):
    q, scored = s["question"], []
    for d in s["documents"]:
        if GRADER.invoke({"question": q, "document": d.page_content}).lower().startswith("yes"):
            scored.append(d)
    print(f"[grader] kept {len(scored)}/{len(s['documents'])}")
    if scored:
        return {"documents": scored, "route": "have_ctx"}
    return {"documents": scored, "route": "need_web"}

def _web(s: State):
    if not web_search:
        return s
    results = web_search.invoke({"query": s["question"]})
    joined = "\n".join(r["content"] for r in results)
    return {"documents": s["documents"] + [Document(page_content=joined)]}

def _generate(s: State):
    return {"generation":
            GEN_PROMPT.invoke({"context": s["documents"],
                               "question": s["question"]})}

# ---  graph ---------------------------------------------------------------
g = StateGraph(State)
g.add_node("_rewrite",   _rewrite)
g.add_node("_retrieve",  _retrieve)
g.add_node("_grade",     _grade)
g.add_node("_web",       _web)
g.add_node("_generate",  _generate)

g.add_edge(START,        "_rewrite")
g.add_edge("_rewrite",   "_retrieve")
g.add_edge("_retrieve",  "_grade")
g.add_conditional_edges("_grade", lambda s: s["route"],
                        {"have_ctx": "_generate",
                         "need_web": "_web"})
g.add_edge("_web",       "_generate")
g.add_edge("_generate",  END)

graph = g.compile()

def answer(q: str) -> str:
    return graph.invoke({"question": q})["generation"]


# 5. FastAPI (+ optional Gradio)

app = FastAPI(title="Greek‑Economy RAG Backend")

try:
    from frontend.ui import demo
    app = gr.mount_gradio_app(app, demo, path="/ui")
    print("[backend] ✅ Gradio UI at /ui")
except Exception:
    print("[backend] ⚠️ No Gradio UI")

class QIn(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json(); q = (data.get("question") or "").strip()
    except Exception:
        q = (await req.body()).decode().strip()
    if not q:
        return {"answer": "Please ask a question about the Greek economy."}
    return {"answer": answer(q)}

@app.get("/")
async def root():
    return {"msg": "Backend running. POST {'question': '…'} to /chat",
            "llm_full": llm_full.model_name,
            "llm_grader": llm_mini.model_name}
