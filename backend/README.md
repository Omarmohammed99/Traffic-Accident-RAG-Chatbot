# Sentinel AI traffic RAG

Flask traffic accident analysis and legal Q&A using Groq for generation, ChromaDB for vector search, and FastEmbed's multilingual E5 model for local embeddings.

## Project layout

```text
RAG/
├── تحديد المتسبب والمتضرر في حوادث المرور.pdf
└── backend/
    ├── main.py
    ├── requirements.txt
    ├── .env                 # local secrets; never commit
    ├── templates/index.html
    └── data/chroma/         # created on first query
```

## Run locally

From `backend/`, install dependencies, copy `.env.example` to `.env`, add your Groq key as `GROQ_API_KEY`, then run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:5000`. The first query downloads/loads the local multilingual embedding model, extracts the PDF text, creates embeddings, and persists a Chroma collection. The Chroma collection is rebuilt when the PDF changes. `GROQ_MODEL` defaults to `llama-3.3-70b-versatile`; set `TRAFFIC_LAW_PDF` to override the PDF path.

## Routes and UI

- `GET /` serves `templates/index.html`.
- `POST /analyze` accepts form fields `party1` and `party2` and renders an evidence-linked accident analysis.
- `POST /chat` accepts JSON `{ "message": "...", "history": [] }` and returns `{ "answer": "...", "sources": [...] }`.
- `GET /health` reports service and PDF availability.

The interface includes an interactive legal assistant sidebar. Both forms support Arabic and English. Legal answers must cite an exact provision found in retrieved excerpts; the assistant is prompted to say when it cannot verify one. PDF page numbers are shown as source locators. This does not establish legal liability or replace review by a qualified professional.

The local multilingual embedding collection is separate from earlier OpenAI-vector collections because vectors from different models cannot be compared. No OpenAI API key is used by this version. For deployment, use a production WSGI server, HTTPS, CSRF protections, request throttling, and a privacy/retention policy for accident statements. Keep `.env` and `data/chroma/` private.
