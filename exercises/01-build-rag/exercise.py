"""
Exercise 1 — Build the RAG Pipeline
=====================================
In this exercise you will assemble the three pieces of a RAG pipeline:
  1. An index  — embed documents, store vectors
  2. A retriever — find the most relevant chunks for a query
  3. A generator — pass question + chunks to the LLM, get an answer

Run:  python exercises/01-build-rag/exercise.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path so we can import from pipeline/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── The document corpus ───────────────────────────────────────────────────
# 25 short movie-review passages. This is what our RAG system retrieves from.

from pipeline.corpus import DOCUMENTS

print(f"Corpus loaded: {len(DOCUMENTS)} documents")
print(f"Example: {DOCUMENTS[0]['text']}\n")


# ── Step 1: Build a search index ─────────────────────────────────────────
#
# We need to turn each document into a vector (a list of numbers) so we can
# find the most similar document for any query.
#
# We use SentenceTransformer — a local embedding model. No API call needed.
# Then FAISS stores the vectors for fast nearest-neighbour search.
#
# TODO: fill in the body of build_index() below.
# The function should:
#   a) encode all document texts using the encoder
#   b) convert to float32 numpy array
#   c) create a FAISS IndexFlatIP index with the right dimension
#   d) normalise and add the embeddings to the index
#   e) return (index, texts, ids)

import numpy as np

def build_index(documents: list[dict]):
    """Return (faiss_index, texts, ids)."""
    from sentence_transformers import SentenceTransformer
    import faiss

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [d["text"] for d in documents]
    ids   = [d["id"]   for d in documents]

    # TODO a: encode texts → embeddings
    embeddings = None  # replace with encoder.encode(...)

    # TODO b: cast to float32
    embeddings = None  # replace with np.array(..., dtype="float32")

    # TODO c: create FAISS index
    dim   = None       # replace with embeddings.shape[1]
    index = None       # replace with faiss.IndexFlatIP(dim)

    # TODO d: normalise and add
    # faiss.normalize_L2(embeddings)
    # index.add(embeddings)

    return index, texts, ids


index, texts, ids = build_index(DOCUMENTS)
print(f"✓ Index built: {index.ntotal if index else '?'} vectors\n")


# ── Step 2: Retrieve relevant chunks ─────────────────────────────────────
#
# Given a question, embed it the same way and find the nearest vectors.
# FAISS returns the top-k most similar documents.
#
# TODO: fill in retrieve() below.
# The function should:
#   a) encode the question
#   b) normalise it
#   c) call index.search(q_emb, top_k) — returns (scores, indices)
#   d) return the corresponding texts and scores

def retrieve(question: str, index, texts: list[str], top_k: int = 3):
    """Return (chunks, scores)."""
    from sentence_transformers import SentenceTransformer
    import faiss

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # TODO a: encode question
    q_emb = None  # replace with encoder.encode([question], ...)

    # TODO b: cast and normalise
    # q_emb = np.array(q_emb, dtype="float32")
    # faiss.normalize_L2(q_emb)

    # TODO c: search
    # scores, indices = index.search(q_emb, top_k)

    # TODO d: return texts and scores
    return [], []


question = "Who directed Inception?"
chunks, scores = retrieve(question, index, texts)
print(f"Q: {question}")
for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
    print(f"  chunk {i} (score {score:.2f}): {chunk[:80]}...")
print()


# ── Step 3: Generate an answer ────────────────────────────────────────────
#
# Pass the question and chunks to the LLM.
# The system prompt says: answer ONLY from the context.
#
# TODO: fill in generate() below.
# The function should:
#   a) format chunks into a numbered context string
#   b) call the Anthropic API with the system prompt and context
#   c) return the text of the response

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer using ONLY the context passages provided.
If the context does not contain enough information, say: "I don't have enough information to answer that."
Be concise — 1-2 sentences."""


def generate(question: str, chunks: list[str]) -> str:
    """Return an answer grounded in the provided chunks."""
    llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # TODO a: build context string
    context = ""  # join chunks with numbering, e.g. "[1] chunk1\n\n[2] chunk2"

    # TODO b: call API
    # r = llm.messages.create(
    #     model="claude-haiku-4-5-20251001",
    #     max_tokens=200,
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    # )

    # TODO c: return answer
    return "TODO"


answer = generate(question, chunks)
print(f"Answer: {answer}")
print()

# ── Verify ────────────────────────────────────────────────────────────────
print("Stuck? See exercises/01-build-rag/solution.py")
