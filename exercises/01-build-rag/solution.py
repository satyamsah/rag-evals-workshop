"""
Exercise 1 — Solution
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import numpy as np
import anthropic

load_dotenv()

from pipeline.corpus import DOCUMENTS


def build_index(documents):
    from sentence_transformers import SentenceTransformer
    import faiss

    encoder  = SentenceTransformer("all-MiniLM-L6-v2")
    texts    = [d["text"] for d in documents]
    ids      = [d["id"]   for d in documents]

    embeddings = encoder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, texts, ids, encoder


def retrieve(question, index, texts, encoder, top_k=3):
    import faiss
    q_emb  = encoder.encode([question], show_progress_bar=False)
    q_emb  = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    chunks = [texts[i] for i in indices[0]]
    sims   = [float(scores[0][j]) for j in range(len(indices[0]))]
    return chunks, sims


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer using ONLY the context passages provided.
If the context does not contain enough information, say: "I don't have enough information to answer that."
Be concise — 1-2 sentences."""


def generate(question, chunks):
    llm     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    r = llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return r.content[0].text.strip()


if __name__ == "__main__":
    index, texts, ids, encoder = build_index(DOCUMENTS)
    print(f"✓ {index.ntotal} vectors indexed\n")

    questions = [
        "Who directed Inception?",
        "How many Oscars did Oppenheimer win?",
        "Who plays Furiosa in Mad Max: Fury Road?",
    ]

    for q in questions:
        chunks, scores = retrieve(q, index, texts, encoder)
        answer = generate(q, chunks)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
