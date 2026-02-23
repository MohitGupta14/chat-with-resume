"""
embeddings.py
=============
CONCEPT: Text Embeddings (FREE — HuggingFace)
----------------------------------------------
Embeddings convert text into numerical vectors that capture MEANING.

Example:
  "I have 5 years of Python experience"  → [0.12, 0.87, -0.34, ...]
  "Python developer with half a decade"  → [0.13, 0.85, -0.33, ...]  ← very similar!
  "I love eating pizza"                  → [-0.54, 0.22, 0.91, ...]  ← very different!

We use HuggingFace's sentence-transformers — 100% FREE, runs locally:
  Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Runs on CPU (no GPU needed)
  - Downloads once (~90MB), then cached locally
  - Great quality for RAG/semantic search

NOTE: Because we changed from OpenAI (1536 dims) to HuggingFace (384 dims),
you MUST set dimension=384 in your Pinecone index.
If you have an old index with 1536 dims, delete it and create a new one!

The vector DB uses these numbers to find "nearest neighbors" —
chunks whose meaning is closest to your question.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Embedding model name — this will be downloaded from HuggingFace Hub
# on first run and cached at ~/.cache/huggingface/
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # ⚠️ Pinecone index must use this dimension!


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace sentence-transformer embeddings instance.
    COMPLETELY FREE — runs locally on your machine.

    This object is used by LangChain in two places:
    1. During INGESTION: embed each chunk before storing in Pinecone
    2. During RETRIEVAL: embed the user's question before searching Pinecone

    Using the SAME model for both is critical — different models produce
    vectors in different "spaces" that can't be compared.

    model_kwargs: {"device": "cpu"} — use CPU (works on any machine)
    encode_kwargs: {"normalize_embeddings": True} — normalize for cosine similarity
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )