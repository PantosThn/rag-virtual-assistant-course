# backend/main.py – REST-only backend for your LangChain RAG pipeline
# ----------------------------------------------------------------------------
#  ▸ Starts a FastAPI server exposing a single POST /chat endpoint
#  ▸ All RAG logic (vector store build, graph compile, answer_question) lives
#    in this file so the front-end can stay completely decoupled.
#
#  Run locally:
#     uvicorn backend.main:app --reload
# ----------------------------------------------------------------------------
#  Environment variables required at runtime (export or put in .env):
#     GROQ_API_KEY        – your Groq API key
#     TAVILY_API_KEY      – your Tavily Search API key
# ----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import List, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain & friends ---------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# 1. Load environment and validate keys
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or export it.")

# ----------------------------------------------------------------------------
# 2. Build vector store from static web docs
# ----------------------------------------------------------------------------
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

print("[backend] Building vector store – may take ~30s on first run …")
_raw_docs: List[Document] = []
for url in URLS:
    _raw_docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250)
_doc_chunks = splitter.split_documents(_raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=_doc_chunks,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# ----------------------------------------------------------------------------
# 3. LLM helper chains (powered by Groq-compatible ChatOpenAI)
# ----------------------------------------------------------------------------
llm_base = ChatOpenAI(
    model_name=GROQ_MODEL,
    temperature=0,
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    model_kwargs={"response_format": {"type": "text"}},
)

# — Retrieval grader ----------------------------------------------------------
GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. Answer strictly with 'yes' or 'no'."),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])
retrieval_grader = GRADER_PROMPT | llm_base | StrOutputParser()

# — Question rewriter --------------------------------------------------------- ---------------------------------------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You rewrite the input question to a version better suited for web search, preserving meaning."),
    ("human", "Original: {question}\nRewritten:"),
])
question_rewriter = REWRITE_PROMPT | llm_base | StrOutputParser()

# — RAG answer generation -----------------------------------------------------
GEN_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context provided. If the context is insufficient, say you don't know. Be concise.\n\n{context}\n\nQuestion: {question}\nAnswer:"""
)
rag_chain = GEN_PROMPT | llm_base | StrOutputParser()

# — Web search fallback -------------------------------------------------------
web_search_tool = TavilySearchResults(k=3)

# ----------------------------------------------------------------------------
# 4. LangGraph workflow
# ----------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def _retrieve(state: GraphState):
    return {"documents": retriever.invoke(state["question"]), "question": state["question"]}


def _grade(state: GraphState):
    q, docs = state["question"], state["documents"]
    relevant, search_flag = [], "No"
    for d in docs:
        if retrieval_grader.invoke({"question": q, "document": d.page_content}).strip().lower().startswith("yes"):
            relevant.append(d)
        else:
            search_flag = "Yes"
    return {"documents": relevant, "question": q, "web_search": search_flag}


def _rewrite(state: GraphState):
    return {"documents": state["documents"], "question": question_rewriter.invoke({"question": state["question"]})}


def _search(state: GraphState):
    q, docs = state["question"], state["documents"]
    results = web_search_tool.invoke({"query": q})
    docs.append(Document(page_content="\n".join(r["content"] for r in results)))
    return {"documents": docs, "question": q}


def _generate(state: GraphState):
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": rag_chain.invoke({"context": state["documents"], "question": state["question"]}),
    }


def _decide(state: GraphState):
    return "_rewrite" if state["web_search"] == "Yes" else "_generate"

workflow = StateGraph(GraphState)
workflow.add_node("_retrieve", _retrieve)
workflow.add_node("_grade", _grade)
workflow.add_node("_rewrite", _rewrite)
workflow.add_node("_search", _search)
workflow.add_node("_generate", _generate)
workflow.add_edge(START, "_retrieve")
workflow.add_edge("_retrieve", "_grade")
workflow.add_conditional_edges("_grade", _decide, {"_rewrite": "_rewrite", "_generate": "_generate"})
workflow.add_edge("_rewrite", "_search")
workflow.add_edge("_search", "_generate")
workflow.add_edge("_generate", END)
_graph = workflow.compile()


def answer_question(question: str) -> str:
    return _graph.invoke({"question": question})["generation"]

# ----------------------------------------------------------------------------
# 5. FastAPI server
# ----------------------------------------------------------------------------
app = FastAPI(title="LangChain RAG Backend (Groq Llama)")

from frontend.ui import demo  # import the Gradio Blocks instance
import gradio as gr

# mount the UI at "/"
app = gr.mount_gradio_app(app, demo, path="/ui")

class QuestionIn(BaseModel):
    question: str

@app.post("/chat")
async def chat(payload: QuestionIn):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return {"answer": answer_question(payload.question)}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}{tb}")

@app.get("/")
async def root():
    return {"msg": f"Backend running. POST JSON → /chat {{\"question\": \"...\"}}. Model: {GROQ_MODEL}"}