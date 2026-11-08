# 🛡️ Sentinel AI — Bilingual Traffic Law RAG & Analysis System

An intelligent, bilingual (Arabic/English) Flask application designed to analyze traffic accident statements and provide precise legal insights using a Retrieval-Augmented Generation (RAG) pipeline. Powered by **Groq** (`openai/gpt-oss-120b`), **ChromaDB** for vector retrieval, and local multilingual embeddings.

> ⚠️ **Disclaimer:** This project provides informational legal analysis only and does not constitute official legal rulings or replace qualified legal professionals.

---

## ✨ Key Features

- **Dual-Party Incident Analysis (`/analyze`):** Compares statements from Party 1 and Party 2 to determine fault and rules of priority based strictly on traffic law excerpts.
- **Interactive Legal Chatbot (`/chat`):** A sidebar assistant handling natural language queries regarding traffic regulations and accident liabilities.
- **Strict Bilingual Support:** Automatically responds in the exact same language as the user's latest input (Arabic or English) without mixing languages.
- **Clean, Citation-Free Responses:** Delivers direct, fluff-free reasoning without cluttering the output with document names, page numbers, or article references.
- **Local & High-Performance Embeddings:** Uses `intfloat/multilingual-e5-small` locally via FastEmbed for fast, accurate semantic search across Arabic and English documents.
- **Automatic Vector Indexing:** Persists a local ChromaDB index and automatically rebuilds it if the underlying PDF changes.

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **AI & LLM:** Groq API (`openai/gpt-oss-120b`)
- **Vector Database & Embeddings:** ChromaDB & Sentence-Transformers (`intfloat/multilingual-e5-small`)
- **Frontend:** HTML5, Tailwind / Custom CSS (Interactive Sidebar Chat & Analysis Form)

---

## 📁 Project Structure

```text
.
├── README.md
├── تحديد المتسبب والمتضرر في حوادث المرور.pdf  # Traffic law knowledge base
└── backend/
    ├── main.py                     # Flask routes and core RAG logic
    ├── requirements.txt            # Python dependencies
    ├── .env.example                # Configuration template
    ├── templates/
    │   └── index.html              # Main UI and sidebar chat interface
    └── data/chroma/                # Local ChromaDB vector store (git-ignored)
🚀 Getting Started LocallyPrerequisitesPython 3.10 or newerA Groq API key (GROQ_API_KEY)Installation & SetupClone the repository and navigate to the project root:Bashgit clone [https://github.com/your-username/Sentinel-Traffic-RAG-Chat.git](https://github.com/your-username/Sentinel-Traffic-RAG-Chat.git)
cd Sentinel-Traffic-RAG-Chat
Set up the backend environment using Conda or Python venv:Bashcd backend
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
Install dependencies:Bashpython -m pip install -r requirements.txt
Configure environment variables:Copy .env.example to .env:BashCopy-Item .env.example .env
Open .env and add your Groq API key:Code snippetGROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
Run the application:Bashpython main.py
Open your browser and navigate to http://127.0.0.1:5000.(Note: The first request will extract the PDF text and initialize the local vector embedding store).🔌 API EndpointsMethodRoutePurposeGET/Render the interactive web interfacePOST/analyzeAccepts party1 and party2 form inputs; returns structured fault analysisPOST/chatAccepts JSON {"message": "...", "history": [...]}; returns assistant responseGET/healthService health-check and PDF availability status⚙️ Configuration VariablesVariableDefaultDescriptionGROQ_API_KEYRequiredYour Groq API access credentialGROQ_MODELopenai/gpt-oss-120bChat completion model on GroqLOCAL_EMBEDDING_MODELintfloat/multilingual-e5-smallMultilingual embedding model for vector searchTRAFFIC_LAW_PDFRoot PDF fileCustom path to the traffic law documentPORT5000Flask server port🔒 Security & Deployment NotesNever commit .env or expose your Groq API keys. The included .gitignore protects virtual environments, vector data, and environment files.For production deployment (e.g., Hugging Face Spaces or Render), configure a WSGI server (like Gunicorn), ensure HTTPS is enabled, and set GROQ_API_KEY securely in your host environment secrets
