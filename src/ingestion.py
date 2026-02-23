"""
ingestion.py
============
CONCEPT: Document Loading & Chunking
--------------------------------------
Before storing text in a vector DB, we need to:
1. LOAD  - Extract raw text from the PDF
2. SPLIT - Break it into small "chunks" (overlapping pieces)

WHY CHUNK? 
- LLMs have token limits, so we can't send the whole resume every time
- Smaller chunks = more precise retrieval
- We find just the relevant piece, not the whole document

WHY OVERLAP?
- If a sentence spans two chunks, overlap ensures context isn't lost
- e.g., chunk_size=500, chunk_overlap=100 means each chunk shares
  100 characters with the next one
"""

from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Document objects.
    Each Document has:
      - page_content: the raw text
      - metadata: {"source": "path/to/file.pdf", "page": 0}
    """
    logger.info(f"Loading PDF from: {file_path}")
    
    # PyPDFLoader is simple and reliable for most resumes
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    logger.info(f"Loaded {len(documents)} pages from PDF")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    RecursiveCharacterTextSplitter tries to split on:
    1. Double newlines (paragraphs) first
    2. Single newlines
    3. Spaces
    4. Individual characters (last resort)
    
    This keeps semantically related text together.
    
    Parameters:
    - chunk_size: max characters per chunk (~125 tokens at ~4 chars/token)
    - chunk_overlap: shared characters between consecutive chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # ~125 tokens per chunk
        chunk_overlap=100,     # overlap to preserve context at boundaries
        separators=["\n\n", "\n", " ", ""],  # try these in order
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enrich metadata for each chunk so we can trace it back later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def load_and_chunk(file_path: str) -> List[Document]:
    """
    Full pipeline: PDF → Documents → Chunks
    This is the main entry point used by vector_store.py
    """
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    
    # Log a preview of the first chunk so you can verify it looks right
    if chunks:
        logger.info(f"Sample chunk:\n{chunks[0].page_content[:200]}...")
    
    return chunks
