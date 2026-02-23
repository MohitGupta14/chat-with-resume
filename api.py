"""
api.py
======
FastAPI Backend
----------------
This is the REST API that sits between Streamlit and the RAG chain.

Endpoints:
  POST /upload      → Upload a PDF resume, ingest into Pinecone
  POST /chat        → Ask a question, get an AI answer
  GET  /health      → Health check
  DELETE /reset     → Delete all vectors for a namespace

Why FastAPI?
- Async support (handles multiple requests efficiently)
- Auto-generates interactive docs at /docs
- Clean Pydantic validation for request/response bodies
- Easy to deploy to any cloud provider later
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import tempfile
import os
import logging

from src.ingestion import load_and_chunk
from src.vector_store import ingest_resume, get_vector_store, delete_namespace
from src.chain import build_rag_chain, ask_question

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chat with Resume API",
    description="Upload a PDF resume and chat with it using AI",
    version="1.0.0"
)

# Allow Streamlit frontend to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
# In production, you'd use Redis or a database
sessions: Dict[str, Dict] = {}


# ─────────────────────────────────────────────
# Request / Response Models (Pydantic)
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    question: str
    session_id: str                          # ties conversation history together
    namespace: str = "default"              # which resume to query

class SourceDocument(BaseModel):
    """A retrieved resume chunk shown as a source citation."""
    text: str
    page: int

class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str
    sources: List[SourceDocument]
    session_id: str

class UploadResponse(BaseModel):
    """Response after uploading a resume."""
    message: str
    namespace: str
    chunks_count: int


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple health check — Streamlit pings this on startup."""
    return {"status": "ok", "message": "Resume Chat API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    namespace: str = "default"
):
    """
    Upload a PDF resume and ingest it into Pinecone.
    
    Steps:
    1. Save uploaded PDF to a temp file
    2. Load & chunk the PDF (ingestion.py)
    3. Embed chunks & store in Pinecone (vector_store.py)
    4. Clean up temp file
    
    The namespace lets us support multiple resumes in one index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save to temp file (UploadFile is a stream, not a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Clear old vectors for this namespace before re-ingesting
        try:
            delete_namespace(namespace)
            logger.info(f"Cleared old vectors for namespace: {namespace}")
        except Exception:
            pass  # Namespace may not exist yet, that's fine
        
        # Chunk the PDF
        chunks = load_and_chunk(tmp_path)
        
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF")
        
        # Ingest into Pinecone
        ingest_resume(chunks, namespace=namespace)
        
        return UploadResponse(
            message=f"✅ Resume '{file.filename}' ingested successfully!",
            namespace=namespace,
            chunks_count=len(chunks)
        )
    
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about the uploaded resume.
    
    Steps:
    1. Load the vector store connection for this namespace
    2. Build the RAG chain
    3. Run the chain with the question + conversation history
    4. Return the answer + source chunks
    
    Session history is stored in memory (sessions dict).
    In production, persist this in Redis with TTL expiry.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get or initialize session
        session_key = f"{request.session_id}:{request.namespace}"
        if session_key not in sessions:
            sessions[session_key] = {"chat_history": []}
        
        chat_history = sessions[session_key]["chat_history"]
        
        # Load vector store and build chain
        vector_store = get_vector_store(namespace=request.namespace)
        chain, retriever = build_rag_chain(vector_store)
        
        # Run the RAG chain
        result = ask_question(
            chain=chain,
            retriever=retriever,
            question=request.question,
            chat_history=chat_history,
        )
        
        # Update session history
        sessions[session_key]["chat_history"] = result["chat_history"]
        
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            session_id=request.session_id,
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.delete("/reset/{namespace}")
async def reset_resume(namespace: str):
    """
    Delete all vectors for a given namespace.
    Useful for replacing a resume without creating duplicates.
    """
    try:
        delete_namespace(namespace)
        # Also clear session history for this namespace
        keys_to_delete = [k for k in sessions if k.endswith(f":{namespace}")]
        for k in keys_to_delete:
            del sessions[k]
        return {"message": f"✅ Namespace '{namespace}' cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("FASTAPI_HOST", "localhost"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=True  # auto-reload on code changes during dev
    )
