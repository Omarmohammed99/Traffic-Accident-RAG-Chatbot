"""Flask RAG application for traffic incident and legal question analysis."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from flask import Flask, jsonify, render_template, request
from groq import Groq
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() 
PDF_NAME = "تحديد المتسبب والمتضرر في حوادث المرور.pdf"
DEFAULT_PDF_PATH = BASE_DIR / PDF_NAME
if not DEFAULT_PDF_PATH.is_file():
    DEFAULT_PDF_PATH = PROJECT_DIR / PDF_NAME
PDF_PATH = Path(os.getenv("TRAFFIC_LAW_PDF", str(DEFAULT_PDF_PATH)))
MAX_STATEMENT_CHARS = 1000
MAX_CHAT_CHARS = 1000
TOP_K = 2
EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
CHROMA_PATH = Path(
    os.getenv(
        "CHROMA_PATH",
        "/tmp/sentinel-traffic-rag/chroma" if os.getenv("VERCEL") else str(BASE_DIR / "data" / "chroma"),
    )
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
groq_client = Groq(api_key=GROQ_API_KEY, timeout=45, max_retries=2) if GROQ_API_KEY else None
embedder: TextEmbedding | None = None
chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma.get_or_create_collection(
    name="traffic_law_multilingual_e5",
    metadata={"hnsw:space": "cosine"},
)


def get_embedder() -> TextEmbedding:
    global embedder
    if embedder is None:
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        if not cache_dir and os.getenv("VERCEL"):
            cache_dir = "/tmp/sentinel-traffic-rag/fastembed"
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache_dir)
    return embedder


def chunks_from_pdf() -> list[dict]:
    """Extract overlapping page-aware chunks to preserve citation locations."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Traffic law PDF not found: {PDF_PATH}")
    reader = PdfReader(str(PDF_PATH))
    chunks: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        words = text.split()
        size, overlap = 180, 35
        for start in range(0, len(words), size - overlap):
            part = " ".join(words[start : start + size]).strip()
            if len(part) >= 80:
                chunks.append({"text": part, "page": page_number})
    return chunks


def ensure_index() -> None:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env.")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Traffic law PDF not found: {PDF_PATH}")
    source_id = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    if collection.count():
        first = collection.get(limit=1, include=["metadatas"])["metadatas"]
        if first and first[0].get("source_id") == source_id:
            return
        collection.delete(where={})
    chunks = chunks_from_pdf()
    if not chunks:
        raise ValueError("No selectable text was extracted from the PDF; it may require OCR.")
    vectorizer = get_embedder()
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset : offset + 64]
        texts = [f"passage: {item['text']}" for item in batch]
        vectors = list(vectorizer.embed(texts))
        collection.add(
            ids=[f"{source_id[:12]}-{offset + i}" for i in range(len(batch))],
            documents=[item["text"] for item in batch],
            embeddings=[vector.tolist() for vector in vectors],
            metadatas=[{"page": item["page"], "source_id": source_id} for item in batch],
        )


def retrieve(query: str) -> list[dict]:
    ensure_index()
    vector = next(get_embedder().embed([f"query: {query}"]))
    result = collection.query(query_embeddings=[vector.tolist()], n_results=min(TOP_K, collection.count()))
    return [
        {"text": text, "page": metadata["page"]}
        for text, metadata in zip(result["documents"][0], result["metadatas"][0])
    ]


def generate_answer(system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env.")
    messages = [{"role": "system", "content": system_prompt}]
    for item in (history or [])[-8:]:
        role, content = item.get("role"), item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content[:MAX_CHAT_CHARS]})
    messages.append({"role": "user", "content": user_prompt})
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


def format_sources(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[Source {i}; PDF page {item['page']}] {item['text']}" for i, item in enumerate(sources, start=1)
    )
LEGAL_SYSTEM_PROMPT = (
    "You are SENTINEL, a careful legal traffic information assistant. "
    "CRITICAL LANGUAGE RULE: You MUST reply in the exact same language as the user's latest message (Arabic if the user writes in Arabic, English if the user writes in English). "
    "Base your legal reasoning strictly on the retrieved traffic-law excerpts. Explicitly cite the exact article number or legal provision shown in the excerpts, in the same language as the answer. "
    "Never invent an article number, penalty, or legal rule. If the excerpts do not establish an article number or enough facts, say that clearly and do not guess. "
    "Do not mention source IDs, document names, or PDF page numbers. "
    "This is informational help, not an official legal opinion."
)


def analyze(party1, party2):
    sources = retrieve(f"Party 1 statement: {party1}\nParty 2 statement: {party2}")
    
  
    prompt = (
        f"PARTY 1 STATEMENT:\n{party1}\n\nPARTY 2 STATEMENT:\n{party2}\n\n"
        f"RETRIEVED TRAFFIC-LAW EXCERPTS:\n{format_sources(sources)}\n\n"
        "Determine whether Party 1 or Party 2 appears responsible based on the statements and excerpts. State the exact article number or provision from the excerpts that supports your conclusion. If the excerpts or facts do not support a reliable decision, say so instead of guessing. Keep the answer concise."
    )
    
  
    verdict = generate_answer(
        LEGAL_SYSTEM_PROMPT + " Give a direct, concise ruling.", 
        prompt
    )
    return {"verdict": verdict, "sources": sources}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze_route():
    party1 = request.form.get("party1", "").strip()
    party2 = request.form.get("party2", "").strip()
    if not party1 or not party2:
        return render_template("index.html", error="Please provide both parties’ statements."), 400
    if len(party1) > MAX_STATEMENT_CHARS or len(party2) > MAX_STATEMENT_CHARS:
        return render_template("index.html", error="Each statement must be 5,000 characters or fewer."), 400
    try:
        result = analyze(party1, party2)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        app.logger.exception("RAG configuration or source error")
        return render_template("index.html", error=str(exc), party1=party1, party2=party2), 503
    except Exception:
        app.logger.exception("Accident analysis failed")
        return render_template("index.html", error="Analysis failed. Check the server configuration and try again.", party1=party1, party2=party2), 502
    return render_template("index.html", party1=party1, party2=party2, result=result)


@app.post("/chat")
def chat_route():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    history = payload.get("history", [])
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Enter a question to continue."), 400
    if len(message) > MAX_CHAT_CHARS:
        return jsonify(error=f"Questions must be {MAX_CHAT_CHARS} characters or fewer."), 400
    if not isinstance(history, list):
        history = []
    try:
        sources = retrieve(message.strip())
        prompt = (
            f"USER'S CURRENT QUESTION:\n{message.strip()}\n\n"
            f"RETRIEVED TRAFFIC-LAW EXCERPTS:\n{format_sources(sources)}\n\n"
            "Answer the current question directly and concisely based on the evidence. Cite the exact article number or provision shown in the excerpts. If no article is shown or the evidence is insufficient, say so rather than inventing a citation. Do not mention source IDs, document names, or PDF page numbers."
        )
        answer = generate_answer(LEGAL_SYSTEM_PROMPT, prompt, history)
        return jsonify(answer=answer, sources=sources)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        app.logger.exception("Chat RAG configuration or source error")
        return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Legal chat failed")
        return jsonify(error="The assistant could not answer right now. Check server configuration and try again."), 502


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "pdf_available": PDF_PATH.is_file()}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
