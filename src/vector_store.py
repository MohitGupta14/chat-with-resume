"""
vector_store.py
===============
CONCEPT: Vector Database (Pinecone)
-------------------------------------
A vector database stores embeddings (number arrays) and lets you search
by SEMANTIC SIMILARITY instead of keyword matching.

Pinecone Architecture:
  - INDEX: like a database table, holds all your vectors
  - NAMESPACE: like a partition, we use one per resume upload
  - VECTOR: a record with {id, values: [float...], metadata: {...}}

How similarity search works:
  1. You send a query vector (embedded question)
  2. Pinecone computes cosine similarity with ALL stored vectors
  3. Returns the top-k most similar vectors
  4. We extract the original text from their metadata

Cosine similarity: measures the angle between two vectors.
  - Score of 1.0 = identical meaning
  - Score of 0.0 = completely unrelated
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from src.embeddings import get_embeddings
from dotenv import load_dotenv
from typing import List
import os
import logging
import time

load_dotenv()
logger = logging.getLogger(__name__)


def get_pinecone_client() -> Pinecone:
    """Initialize and return Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env file")
    return Pinecone(api_key=api_key)


def ensure_index_exists(pc: Pinecone, index_name: str) -> None:
    """
    Create the Pinecone index if it doesn't exist yet.
    
    dimension=1536 must match OpenAI's text-embedding-3-small output size.
    metric="cosine" is best for text similarity tasks.
    
    ServerlessSpec uses Pinecone's free serverless tier (no pods needed).
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,          # must match embedding model output
            metric="cosine",         # cosine similarity for text
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"   # free tier region
            )
        )
        # Wait for index to be ready (usually 10-30 seconds)
        logger.info("Waiting for index to be ready...")
        time.sleep(20)
    else:
        logger.info(f"Index '{index_name}' already exists, skipping creation")


def ingest_resume(chunks: List[Document], namespace: str = "default") -> PineconeVectorStore:
    """
    INGESTION PIPELINE:
    chunks → embed each chunk → upsert vectors into Pinecone
    
    namespace: lets you store multiple resumes in the same index
    by separating them into isolated partitions.
    
    LangChain's PineconeVectorStore.from_documents() handles:
    - Calling the embedding model for each chunk
    - Formatting vectors for Pinecone's API
    - Upserting in batches (100 vectors at a time)
    """
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", "resume-chat")
    
    ensure_index_exists(pc, index_name)
    
    logger.info(f"Ingesting {len(chunks)} chunks into namespace '{namespace}'")
    
    embeddings = get_embeddings()
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
    )
    
    logger.info("Ingestion complete!")
    return vector_store


def get_vector_store(namespace: str = "default") -> PineconeVectorStore:
    """
    Connect to an EXISTING Pinecone index for querying.
    Used at query time — we don't re-ingest, just load the connection.
    """
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", "resume-chat")
    embeddings = get_embeddings()
    
    return PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings,
        namespace=namespace,
    )


def delete_namespace(namespace: str) -> None:
    """
    Delete all vectors in a namespace (e.g., when re-uploading a resume).
    This avoids duplicate vectors from multiple uploads of the same resume.
    """
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", "resume-chat")
    index = pc.Index(index_name)
    index.delete(delete_all=True, namespace=namespace)
    logger.info(f"Deleted namespace: {namespace}")
