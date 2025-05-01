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
#     OPENAI_API_KEY   – OpenAI key
#     TAVILY_API_KEY   – Tavily Search key
# ----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # ✅ use Pydantic v2 directly

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# 1. Load .env and validate keys
# ----------------------------------------------------------------------------

os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("TAVILY_API_KEY", "")

ROOT_DIR = Path(__file__).resolve().parent.parent  # → project root
load_dotenv(ROOT_DIR / ".env")



# ----------------------------------------------------------------------------
# 2. Build vector store (one-time, in-memory)
# ----------------------------------------------------------------------------
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

print("[backend] Building vector store – occurs only once at startup …")
_raw_docs = []
for url in URLS:
    _raw_docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)
_doc_splits = splitter.split_documents(_raw_docs)

vectorstore = Chroma.from_documents(
    documents=_doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# ----------------------------------------------------------------------------
# 3. LLM helper chains (grader, re-writer, generator)
# ----------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str

# — Retrieval grader ----------------------------------------------------------
SYSTEM_GRADER = (
    "You are a grader assessing relevance of a retrieved document to a user "
    "question. If the document contains keyword(s) or semantic meaning related "
    "to the question, grade it as relevant. Give a binary 'yes' or 'no'."
)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_GRADER),
        ("human", "Retrieved document:\n\n {document} \n\n User question: {question}"),
    ]
)

llm_grader = (
    ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    .with_structured_output(GradeDocuments)
)
retrieval_grader = grade_prompt | llm_grader

# — Question re-writer --------------------------------------------------------
REWRITE_SYSTEM = (
    "You are a question re-writer that converts an input question into a better "
    "version optimized for web search."
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM),
        (
            "human",
            "Here is the initial question:\n\n{question}\nFormulate an improved question.",
        ),
    ]
)

question_rewriter = (
    rewrite_prompt
    | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    | StrOutputParser()
)

# — RAG generator -------------------------------------------------------------
GEN_PROMPT = ChatPromptTemplate.from_template(
    """Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}\nAnswer concisely:"""
)
rag_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = GEN_PROMPT | rag_llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

# ----------------------------------------------------------------------------
# 4. LangGraph state machine
# ----------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def _retrieve(state: GraphState):
    q = state["question"]
    return {"documents": retriever.get_relevant_documents(q), "question": q}


def _grade(state: GraphState):
    q, docs = state["question"], state["documents"]
    filtered: List[Document] = []
    need_search = "No"
    for d in docs:
        if retrieval_grader.invoke({"question": q, "document": d.page_content}).binary_score == "yes":
            filtered.append(d)
        else:
            need_search = "Yes"
    return {"documents": filtered, "question": q, "web_search": need_search}


def _rewrite(state: GraphState):
    return {
        "documents": state["documents"],
        "question": question_rewriter.invoke({"question": state["question"]}),
    }


def _search(state: GraphState):
    q, docs = state["question"], state["documents"]
    results = web_search_tool.invoke({"query": q})
    docs.append(Document(page_content="\n".join(r["content"] for r in results)))
    return {"documents": docs, "question": q}


def _generate(state: GraphState):
    answer = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": answer}


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
app = FastAPI(title="LangChain RAG Backend")

class QuestionIn(BaseModel):
    question: str

@app.post("/chat")
async def chat(payload: QuestionIn):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return {"answer": answer_question(payload.question)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/")
async def root():
    return {"msg": "Backend running. POST JSON → /chat {\"question\": \"...\"}"}