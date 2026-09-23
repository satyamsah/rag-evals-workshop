"""
Exercise 1 — Build the RAG Pipeline
=====================================
You will build the three pieces of a RAG pipeline by uncommenting lines
one step at a time. After each step, run the file and see what changed.

Run:  python exercises/01-build-rag/exercise.py
"""

import os
import sys
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import anthropic

load_dotenv()

from pipeline.corpus import DOCUMENTS

# Encoder loaded once at the top — both functions below share it
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Corpus loaded: {len(DOCUMENTS)} documents")
print(f"Example: {DOCUMENTS[0]['text'][:80]}...\n")


# ── Step 1: Build a search index ──────────────────────────────────────────
#
# We convert every document into a vector (a list of numbers).
# Documents with similar meaning will have similar vectors.
# FAISS stores those vectors so we can search them instantly.
#
# Uncomment each line below one at a time and re-run after each one.
#

def build_index(documents):
    texts = [d["text"] for d in documents]

    # Step 1a — convert every document into a vector
    # embeddings = encoder.encode(texts, show_progress_bar=False)

    # Step 1b — FAISS needs float32 numbers
    # embeddings = np.array(embeddings, dtype="float32")

    # Step 1c — create the index (IndexFlatIP = cosine similarity search)
    # index = faiss.IndexFlatIP(embeddings.shape[1])

    # Step 1d — normalise vectors and add them to the index
    # faiss.normalize_L2(embeddings)
    # index.add(embeddings)

    # Step 1e — return everything we need later
    # return index, texts

    # TODO: REMOVE THE LINE BELOW ONCE ALL STEPS ABOVE ARE UNCOMMENTED
    return None, texts


index, texts = build_index(DOCUMENTS)
if index is not None:
    print(f"✓ Index built: {index.ntotal} vectors stored\n")
else:
    print("⚠ Index not built yet — complete Step 1 first\n")


# ── Step 2: Retrieve relevant chunks ─────────────────────────────────────
#
# Given a question, we embed it the same way as the documents.
# Then FAISS finds the 3 most similar document vectors — our top-k chunks.
#

def retrieve(question, index, texts, top_k=3):
    if index is None:
        return [], []  # Step 1 not done yet

    # Step 2a — embed the question the same way we embedded documents
    # q_emb = encoder.encode([question], show_progress_bar=False)

    # Step 2b — normalise and search the index
    # q_emb = np.array(q_emb, dtype="float32")
    # faiss.normalize_L2(q_emb)
    # scores, indices = index.search(q_emb, top_k)

    # Step 2c — return the matching texts and their similarity scores
    # chunks = [texts[i] for i in indices[0]]
    # sims   = [float(scores[0][j]) for j in range(len(indices[0]))]
    # return chunks, sims

    # TODO: REMOVE THE LINE BELOW ONCE ALL STEPS ABOVE ARE UNCOMMENTED
    return [], []


question = "Who directed Inception?"
chunks, scores = retrieve(question, index, texts)
if chunks:
    print(f"Q: {question}")
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        print(f"  chunk {i} (score {score:.2f}): {chunk[:80]}...")
    print()
else:
    print("⚠ No chunks retrieved — complete Step 2 first\n")


# ── Step 3: Generate an answer ────────────────────────────────────────────
#
# We pass the question AND the retrieved chunks to Claude.
# The system prompt tells Claude: answer ONLY from what is in the context.
# This is the grounding instruction — it is what controls faithfulness.
#

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer using ONLY the context passages provided.
If the context does not contain enough information, say: "I don't have enough information to answer that."
Be concise — 1 sentence."""


def generate(question, chunks):
    if not chunks:
        return "⚠ No context — complete Steps 1 and 2 first"

    llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Step 3a — format the chunks into a numbered context block
    # context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

    # Step 3b — call Claude with the question and context
    # r = llm.messages.create(
    #     model="claude-haiku-4-5-20251001",
    #     max_tokens=200,
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    # )

    # Step 3c — return the answer text
    # return r.content[0].text.strip()

    # TODO: REMOVE THE LINE BELOW ONCE ALL STEPS ABOVE ARE UNCOMMENTED
    return "⚠ No answer yet — complete Step 3 first"


answer = generate(question, chunks)
print(f"Answer: {answer}\n")


# ── When everything is uncommented you should see: ────────────────────────
#
# ✓ Index built: 24 vectors stored
#
# Q: Who directed Inception?
#   chunk 1 (score 0.75): Inception is a 2010 sci-fi thriller directed by Christopher Nolan...
# Answer: Christopher Nolan directed Inception.
