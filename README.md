# 📄 Chat with Resume

An AI-powered app to chat with any PDF resume — **100% FREE** using **Groq (LLaMA-3)**, **HuggingFace Embeddings**, **Pinecone**, **FastAPI**, and **Streamlit**.

---

## 💸 Cost: $0 — Here's Why

| Component | Provider | Cost |
|---|---|---|
| LLM (chat) | Groq — LLaMA-3.3 70B | Free tier |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, runs locally |
| Vector DB | Pinecone Serverless | Free tier (1 index) |
| Backend | FastAPI | Free (open source) |
| Frontend | Streamlit | Free (open source) |

---

## 🏗️ Architecture

```
┌────────────┐     HTTP      ┌─────────────┐   LangChain   ┌──────────────────┐
│  Streamlit │ ────────────► │   FastAPI   │ ────────────► │  Groq LLaMA-3    │
│  (app.py)  │ ◄──────────── │   (api.py)  │               │  (FREE, fast!)   │
└────────────┘               └─────────────┘               └──────────────────┘
                                    │                                ▲
                          Pinecone SDK               relevant chunks │
                                    ▼                                │
                             ┌─────────────┐   HuggingFace Embeddings│
                             │  Pinecone   │ ──────────────────────► │
                             │ Vector DB   │   (local, no API key)
                             └─────────────┘
```

## 🗂️ Project Structure

```
chat-with-resume/
├── api.py                  # FastAPI backend (REST API)
├── app.py                  # Streamlit frontend (UI)
├── src/
│   ├── ingestion.py        # PDF loading & chunking
│   ├── embeddings.py       # HuggingFace embeddings (FREE, local)
│   ├── vector_store.py     # Pinecone CRUD operations
│   └── chain.py            # LangChain RAG pipeline with Groq
├── .env.example
└── requirements.txt
```

---

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run will download the HuggingFace embedding model (~90MB). This is cached after that.

### 2. Get FREE API Keys

**Groq (LLM — Free):**
1. Go to https://console.groq.com/keys
2. Sign up (free, no credit card)
3. Click "Create API Key"
4. Copy key starting with `gsk_...`

**Pinecone (Vector DB — Free tier):**
1. Go to https://app.pinecone.io
2. Sign up free (1 free index)
3. Go to API Keys → Copy your key
4. Index is **auto-created** on first run with correct settings

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_your-key-here
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=resume-chat
```

> ✅ No OpenAI key needed!

### 4. Run the App

Open **two terminals**:

**Terminal 1 — Backend:**
```bash
python api.py
```

**Terminal 2 — Frontend:**
```bash
streamlit run app.py
```

Open http://localhost:8501 🎉

---

## 💬 How to Use

1. Upload a PDF resume in the sidebar
2. Click **"⚡ Ingest Resume"** (takes ~15s — embedding happens locally)
3. Ask questions in the chat!

**Example questions:**
- "Summarize this person's work experience"
- "What programming languages do they know?"
- "Are they a good fit for a backend role?"
- "What was their most impactful project?"

---

## 🧠 Concepts Explained

### RAG (Retrieval-Augmented Generation)
1. **Store** resume chunks as vectors in Pinecone
2. **Retrieve** the most relevant chunks for each question
3. **Generate** an answer using those chunks as context

### Chunking
PDFs are split into overlapping ~500-character chunks (100-char overlap). Preserves context at section boundaries.

### Embeddings (HuggingFace — Local & Free)
`all-MiniLM-L6-v2` converts text → 384-dimensional vectors that capture meaning. Runs on CPU, no internet required after first download.

### Groq + LLaMA-3
Groq is a cloud provider running open-source LLaMA-3 models at extremely high speed (hundreds of tokens/second). Free tier is generous for development.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check if API is running |
| `POST` | `/upload` | Upload & ingest a PDF resume |
| `POST` | `/chat` | Ask a question |
| `DELETE` | `/reset/{namespace}` | Clear all vectors for a namespace |

Interactive docs: http://localhost:8000/docs

---

## ⚠️ Important: Pinecone Index Dimension

This version uses **384 dimensions** (HuggingFace), not 1536 (OpenAI).

If you previously had an OpenAI-based index, **delete it** from the Pinecone dashboard before running. The app will auto-create a new 384-dim index.

---

## 🔧 Customization

**Change the LLM model** in `src/chain.py`:
```python
model="llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" (even faster)
```

**Change chunk size** in `src/ingestion.py`:
```python
chunk_size=500    # larger = more context per chunk
chunk_overlap=100
```

**Retrieve more chunks** in `src/chain.py`:
```python
search_kwargs={"k": 4}  # increase for broader context
```