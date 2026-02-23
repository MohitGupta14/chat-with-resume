"""
chain.py
========
CONCEPT: RAG Chain (Retrieval-Augmented Generation)
-----------------------------------------------------
This is the heart of the project. The RAG chain connects:
  1. RETRIEVER  → finds relevant resume chunks from Pinecone
  2. PROMPT     → formats chunks + question into a structured prompt
  3. LLM        → generates a natural language answer
  4. PARSER     → extracts clean text from the LLM response

LangChain Expression Language (LCEL):
We use the pipe `|` operator to chain steps:
  retriever | prompt | llm | parser

Each step's output becomes the next step's input.

Conversation Memory:
We maintain a chat_history list so the LLM can understand follow-up
questions like "Tell me more about that" or "What else did he do there?"

BUG FIX NOTE:
The retriever and HuggingFace embedder expect a plain STRING as input,
not a dict. So we use RunnableLambda(lambda x: x["question"]) to extract
just the question string before it reaches the retriever.
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema import HumanMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI assistant that answers questions about a candidate's resume.
You have access to the resume content below. Answer questions accurately and helpfully.

Guidelines:
- Be specific and cite details from the resume when possible
- If asked about skills, experience, or education — pull exact details
- If the resume doesn't contain the answer, say "This information isn't in the resume"
- Keep answers concise but thorough
- Speak as if you are a helpful recruiter who knows this candidate well

Resume Content (retrieved relevant sections):
{context}
"""

QUESTION_TEMPLATE = """Based on the resume information above, please answer this question:
{question}"""


def format_docs(docs) -> str:
    """
    Convert retrieved Document objects into a single formatted string.
    This string is injected into the {context} slot of the prompt.
    """
    if not docs:
        return "No relevant sections found in the resume."

    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Section {i} - Page {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vector_store: PineconeVectorStore):
    """
    Build the full RAG chain using LangChain Expression Language (LCEL).

    Chain flow:
    ┌─────────────────────────────────────────────────────┐
    │ Input dict: {"question": str, "chat_history": [...]} │
    │    ↓                                                 │
    │ Extract question string (RunnableLambda)             │
    │    ↓                                                 │
    │ Retriever → search Pinecone → top 4 chunks          │
    │    ↓                                                 │
    │ format_docs() → combine chunks into context string  │
    │    ↓                                                 │
    │ ChatPromptTemplate → fill {context}, {question}     │
    │    ↓                                                 │
    │ ChatGroq (LLaMA-3.3 70B) → generate response       │
    │    ↓                                                 │
    │ StrOutputParser → return plain string               │
    └─────────────────────────────────────────────────────┘

    KEY FIX: The retriever receives x["question"] (a string), NOT the
    full dict. HuggingFace embeddings crash if given a dict.
    """

    # RETRIEVER: search Pinecone for top-k similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # LLM: Groq runs LLaMA-3.3 70B — free and very fast
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", QUESTION_TEMPLATE),
    ])

    # OUTPUT PARSER
    parser = StrOutputParser()

    # ── THE FIX ──────────────────────────────────────────────────────────
    # chain.invoke() receives a dict like:
    #   {"question": "What skills does he have?", "chat_history": [...]}
    #
    # The retriever needs ONLY the question string — not the whole dict.
    # We use RunnableLambda(lambda x: x["question"]) to extract it.
    # Without this, HuggingFace's embed_query() receives a dict and crashes
    # with: AttributeError: 'dict' object has no attribute 'replace'
    # ─────────────────────────────────────────────────────────────────────
    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
        | parser
    )

    return chain, retriever


def ask_question(
    chain,
    retriever,
    question: str,
    chat_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with source chunks.

    Returns:
    {
        "answer": "John has 5 years of Python experience at...",
        "sources": [{"text": "...", "page": 0}, ...],
        "chat_history": [...updated history...]
    }
    """
    if chat_history is None:
        chat_history = []

    # Convert history dicts to LangChain message objects
    lc_history = []
    for msg in chat_history:
        if msg["role"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            lc_history.append(AIMessage(content=msg["content"]))

    # Run the chain — input is a clean dict with string question
    answer = chain.invoke({
        "question": question,        # ← plain string, extracted in chain
        "chat_history": lc_history,
    })

    # Retrieve sources separately to show citations in UI
    source_docs = retriever.invoke(question)
    sources = [
        {
            "text": doc.page_content,
            "page": doc.metadata.get("page", 0),
        }
        for doc in source_docs
    ]

    # Update conversation history for next turn
    updated_history = chat_history + [
        {"role": "human", "content": question},
        {"role": "ai", "content": answer},
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chat_history": updated_history,
    }