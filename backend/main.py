"""Flask RAG application for traffic incident and legal question analysis."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

# Vercel's deployment image is read-only. FastEmbed's own cache_dir does not
# control Hugging Face Hub/Xet caches, which otherwise default under $HOME.
# Set these before importing FastEmbed (and huggingface_hub transitively).
if os.getenv("VERCEL"):
    _runtime_cache = Path("/tmp/sentinel-traffic-rag")
    _hf_home = _runtime_cache / "huggingface"
    _fastembed_cache = _runtime_cache / "fastembed"
    os.environ["HF_HOME"] = str(_hf_home)
    # FastEmbed already uses its own HF snapshot cache. Keep the Hub cache on
    # that same volume and disable Xet's reconstruction path, which needs extra
    # temporary disk space and exceeds Vercel's 500 MB /tmp limit for this model.
    os.environ["HF_HUB_CACHE"] = str(_fastembed_cache)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CACHE"] = str(_fastembed_cache / "xet")
    os.environ["XDG_CACHE_HOME"] = str(_runtime_cache / "xdg")
    os.environ["FASTEMBED_CACHE_PATH"] = str(_fastembed_cache)
    # Remove large, incomplete snapshots left by earlier FastEmbed/Hub attempts.
    shutil.rmtree(_hf_home, ignore_errors=True)
    _fastembed_cache.mkdir(parents=True, exist_ok=True)
    for _old_cache in _fastembed_cache.iterdir():
        if _old_cache.name == "vercel-model":
            for _partial in _old_cache.rglob("*.partial"):
                _partial.unlink(missing_ok=True)
        elif _old_cache.is_dir():
            shutil.rmtree(_old_cache, ignore_errors=True)
        else:
            _old_cache.unlink(missing_ok=True)

    # Force the already-imported Hub constants too. Some Vercel Python
    # bootstrap paths import huggingface_hub before this module's env setup.
    from huggingface_hub import constants as _hf_constants

    _hf_constants.HF_HUB_DISABLE_XET = True

import chromadb
from fastembed import TextEmbedding
from flask import Flask, jsonify, render_template, request
from groq import Groq
from pypdf import PdfReader
import requests
from urllib.parse import quote

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
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    DEFAULT_EMBEDDING_MODEL,
)
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
active_embedding_model: str | None = None
chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma.get_or_create_collection(
    name="traffic_law_multilingual_minilm_v1",
    metadata={"hnsw:space": "cosine"},
)


def get_embedder() -> TextEmbedding:
    global embedder, active_embedding_model
    if embedder is None:
        supported_models = {
            item["model"] for item in TextEmbedding.list_supported_models()
        }
        model_name = EMBEDDING_MODEL
        if model_name not in supported_models:
            if DEFAULT_EMBEDDING_MODEL not in supported_models:
                raise RuntimeError(
                    "The configured embedding model and the multilingual fallback "
                    "are not supported by this FastEmbed version."
                )
            app.logger.warning(
                "FastEmbed does not support LOCAL_EMBEDDING_MODEL=%r; using %r.",
                model_name,
                DEFAULT_EMBEDDING_MODEL,
            )
            model_name = DEFAULT_EMBEDDING_MODEL
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        if os.getenv("VERCEL"):
            cache_dir = cache_dir or "/tmp/sentinel-traffic-rag/fastembed"
            model_description = next(
                item for item in TextEmbedding.list_supported_models()
                if item["model"].lower() == model_name.lower()
            )
            model_dir = Path(cache_dir) / "vercel-model"
            download_model_files_directly(model_description, model_dir)
            embedder = TextEmbedding(
                model_name=model_name,
                specific_model_path=str(model_dir),
            )
        else:
            embedder = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        active_embedding_model = model_name
    return embedder


def download_model_files_directly(model: dict, model_dir: Path) -> None:
    """Stream required model assets straight to /tmp without Hub/Xet staging."""
    source_repo = (model.get("sources") or {}).get("hf")
    model_file = model.get("model_file")
    if not source_repo or not model_file:
        raise RuntimeError("The configured FastEmbed model has no Hugging Face source.")

    listing_url = f"https://huggingface.co/api/models/{source_repo}/tree/main?recursive=true&expand=false"
    listing_response = requests.get(listing_url, timeout=(15, 45))
    listing_response.raise_for_status()
    available = {
        entry["path"]: entry
        for entry in listing_response.json()
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }

    required_names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        model_file,
        *model.get("additional_files", []),
    }
    required_files = [name for name in required_names if name in available]
    if model_file not in required_files:
        raise RuntimeError(f"Model file {model_file!r} is missing from {source_repo}.")

    model_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(available[name].get("size", 0) for name in required_files)
    available_bytes = shutil.disk_usage(model_dir).free
    app.logger.info(
        "Embedding cache has %.1f MiB free; required assets total %.1f MiB.",
        available_bytes / (1024 * 1024),
        required_bytes / (1024 * 1024),
    )
    if available_bytes < required_bytes:
        raise RuntimeError(
            "Vercel has insufficient temporary storage for the embedding model. "
            "Clear old function cache files or use a persistent host."
        )

    with requests.Session() as session:
        for filename in required_files:
            file_info = available[filename]
            destination = model_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected_size = file_info.get("size")
            if destination.is_file() and destination.stat().st_size == expected_size:
                continue

            destination.unlink(missing_ok=True)
            partial_path = destination.with_name(destination.name + ".partial")
            partial_path.unlink(missing_ok=True)
            url = f"https://huggingface.co/{source_repo}/resolve/main/{quote(filename, safe='/')}"
            try:
                with session.get(url, stream=True, timeout=(15, 180)) as response:
                    response.raise_for_status()
                    with partial_path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if expected_size is not None and partial_path.stat().st_size != expected_size:
                    raise RuntimeError(f"Downloaded model asset {filename!r} is incomplete.")
                partial_path.replace(destination)
            except Exception:
                partial_path.unlink(missing_ok=True)
                raise


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
        texts = [item["text"] for item in batch]
        vectors = list(vectorizer.embed(texts))
        collection.add(
            ids=[f"{source_id[:12]}-{offset + i}" for i in range(len(batch))],
            documents=[item["text"] for item in batch],
            embeddings=[vector.tolist() for vector in vectors],
            metadatas=[{"page": item["page"], "source_id": source_id} for item in batch],
        )


def retrieve(query: str) -> list[dict]:
    ensure_index()
    vector = next(get_embedder().embed([query]))
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
    return {
        "status": "ok",
        "model": MODEL,
        "embedding_model": active_embedding_model or EMBEDDING_MODEL,
        "pdf_available": PDF_PATH.is_file(),
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
