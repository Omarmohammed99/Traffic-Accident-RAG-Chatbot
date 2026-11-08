# Sentinel AI — Traffic Law RAG

A bilingual (Arabic/English) Flask application for reviewing two traffic incident statements and asking questions about the included Arabic traffic-law PDF. Groq generates answers; ChromaDB retrieves relevant passages; FastEmbed creates multilingual embeddings locally.

> This project provides informational analysis only. It does not make official findings of fault or replace a qualified legal professional.

## Features

- Compare Party 1 and Party 2 accident statements.
- Ask traffic-law questions in the interactive chat sidebar.
- Retrieve supporting passages from the traffic-law PDF and show their PDF page numbers.
- Respond in the user's language and require exact article/provision citations from retrieved evidence. If a citation cannot be verified, the assistant should say so rather than invent one.
- Persist a local ChromaDB index; rebuild it automatically when the PDF changes.
- Use Groq for language generation and a local multilingual embedding model for retrieval. No OpenAI service or key is required.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access on first install and first embedding-model download

## Run locally

In PowerShell, from the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set `GROQ_API_KEY` to your key. The default model is configured as:

```dotenv
GROQ_MODEL=llama-3.3-70b-versatile
```

Then start the app:

```powershell
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The first request extracts PDF text and builds the local vector index; it may take longer while the embedding model is downloaded. Put the traffic-law PDF in the repository root as `تحديد المتسبب والمتضرر في حوادث المرور.pdf`, or set `TRAFFIC_LAW_PDF` in `backend/.env` to another file path.

## Project structure

```text
.
├── README.md
├── تحديد المتسبب والمتضرر في حوادث المرور.pdf
└── backend/
    ├── main.py                 # Flask routes and RAG pipeline
    ├── requirements.txt
    ├── .env.example            # safe configuration template
    ├── templates/
    │   └── index.html          # Sentinel UI and chat sidebar
    └── data/chroma/            # generated local vector index (ignored by Git)
```

## API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Render the application |
| `POST` | `/analyze` | Accept form fields `party1` and `party2`; render an evidence-linked incident analysis |
| `POST` | `/chat` | Accept JSON `{ "message": "...", "history": [] }`; return `{ "answer": "...", "sources": [...] }` |
| `GET` | `/health` | Return service and PDF availability |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Groq API credential |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat-completion model |
| `LOCAL_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local Arabic/English embedding model |
| `TRAFFIC_LAW_PDF` | PDF in repository root | Optional path to the law document |
| `PORT` | `5000` | Flask development server port |

Never commit `.env`. The included `.gitignore` excludes environment files, virtual environments, and generated vector data. If a key has been exposed, revoke it in the provider dashboard and replace it with a new key.

## Deployment notes

Publishing this repository to GitHub does not host the Flask application. For a live deployment, use a production WSGI server and configure `GROQ_API_KEY` as a secret in the hosting provider. Use HTTPS, protect the chat route against abuse, configure CSRF protections if browser session features are added, and decide how accident statements and logs should be retained. Keep the vector database private. Verify the PDF text extraction and legal citations with a qualified reviewer before relying on the application in real cases.
