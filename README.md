# 🛡️ Sentinel AI — Bilingual Traffic Law RAG & Analysis System

A bilingual (Arabic/English) Flask application that analyzes two traffic accident statements and answers traffic-law questions using retrieval-augmented generation (RAG). Groq powers the language model and translates English queries to Arabic. ChromaDB stores searchable law passages using compact deterministic hashed-token vectors; original Arabic excerpts remain the citation source. This avoids downloading a large embedding model into Vercel's limited temporary filesystem.

> ⚠️ **Disclaimer:** This project provides informational legal analysis only. It does not make official rulings or replace advice from a qualified legal professional.

## ✨ Key Features

- **Dual-party incident analysis (`/analyze`):** Compares Party 1 and Party 2 statements against retrieved passages from the traffic-law PDF and explains the supported conclusion and uncertainty.
- **Interactive legal chatbot (`/chat`):** Answers natural-language questions about traffic regulations using the same PDF-backed RAG pipeline.
- **Arabic and English support:** Accepts either language and responds in the language of the latest user input. The interface supports Arabic right-to-left text.
- **Evidence-based legal citations:** Legal conclusions must identify the exact article or provision number found in retrieved PDF text. The assistant is instructed to say when it cannot verify a citation instead of inventing one. Retrieved excerpts include PDF page locators.
- **Serverless-friendly retrieval:** Deterministic hashed Arabic token and phrase vectors require no model download; Groq translates English search queries to Arabic while indexed legal excerpts remain in the source Arabic.
- **Persistent vector index:** ChromaDB stores the local index and rebuilds it when the source PDF changes.
- **Sentinel dark UI:** Futuristic Tailwind interface with an incident form and interactive legal assistant sidebar.

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **AI model:** Groq API, default `openai/gpt-oss-120b`
- **Vector store:** ChromaDB
- **Embeddings:** Deterministic 4,096-dimension hashed lexical vectors; Groq-assisted English-to-Arabic query translation
- **Frontend:** Jinja2, HTML, Tailwind CSS, custom CSS, and JavaScript

## 📁 Project Structure

```text
.
├── README.md
├── تحديد المتسبب والمتضرر في حوادث المرور.pdf  # Traffic-law knowledge base
└── backend/
    ├── main.py                     # Flask routes and RAG pipeline
    ├── requirements.txt            # Python dependencies
    ├── .env.example                # Configuration template
    ├── README.md                   # Backend setup notes
    ├── templates/
    │   └── index.html              # Sentinel UI and chat sidebar
    └── data/chroma/                # Local ChromaDB index (Git-ignored)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- A Groq API key
- Internet access for dependency installation and Groq API requests

### Install and run (PowerShell)

Clone the repository and enter the backend folder:

```powershell
git clone https://github.com/Omarmohammed99/Traffic-Accident-RAG-Chatbot.git
cd Traffic-Accident-RAG-Chatbot/backend
```

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create your local environment file and set your Groq key:

```powershell
Copy-Item .env.example .env
```

Edit `backend/.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Start Flask:

```powershell
python main.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The first RAG request extracts text from the PDF and builds the ChromaDB index; no embedding model is downloaded. The PDF is included at the repository root; set `TRAFFIC_LAW_PDF` in `backend/.env` to use a different file.

## 🔌 API Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Render the application interface |
| `POST` | `/analyze` | Accept `party1` and `party2` form fields and render an evidence-linked incident analysis |
| `POST` | `/chat` | Accept JSON `{ "message": "...", "history": [] }` and return `{ "answer": "...", "sources": [...] }` |
| `GET` | `/health` | Report service status, model, and PDF availability |

## ⚙️ Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Required | Groq API credential |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat-completion model |
| `TRAFFIC_LAW_PDF` | Traffic PDF in repository root | Optional path to the law document |
| `PORT` | `5000` | Flask development-server port |

## 🔒 Security and Deployment

- Never commit `.env` or expose Groq API keys. `.gitignore` excludes environment files, virtual environments, and generated ChromaDB data.
- For production, use a WSGI server such as Gunicorn, enable HTTPS, add request throttling and appropriate CSRF protections, and set `GROQ_API_KEY` as a hosting-provider secret.
- Decide how incident statements and logs are retained. Keep the vector database private.
- Confirm that PDF text extraction works and have legal citations reviewed before relying on results in real cases.
