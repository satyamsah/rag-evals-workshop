"""
pipeline/rag.py — the RAG pipeline used throughout the workshop.

This is the COMPLETE, working version. Exercises ask you to build parts of it
step-by-step. Come back here if you need to see how a piece fits together.

Architecture:
  build_index()    — embeds documents with a sentence transformer, stores in FAISS
  retrieve()       — finds the top-k most similar chunks for a query
  generate()       — passes question + chunks to Claude, returns an answer

Run the full pipeline end-to-end:
  python pipeline/rag.py

── What to watch for ────────────────────────────────────────────────────────

  EMBEDDING MODEL (line ~50)
    We use a tiny local model (all-MiniLM-L6-v2) — no API cost, instant setup.
    In production you would use text-embedding-3-small (OpenAI) or equivalent.
    The retrieval logic is identical either way.

  RETRIEVAL (retrieve function)
    Returns (chunks, scores). Higher score = more similar.
    Watch what happens when the question uses different words than the document.
    That is the vocabulary mismatch problem — and it shows up in your eval scores.

  GENERATION (generate function)
    Strict system prompt: answer ONLY from the provided context.
    This is intentional. It means hallucination shows up as "I don't know"
    rather than a confident wrong answer — which is easier to catch in evals.
"""

import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

import numpy as np
import anthropic

load_dotenv()

# ── Lazy imports (heavy) ──────────────────────────────────────────────────
# Only import when needed so check_setup.py stays fast.

def _get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def _get_faiss():
    import faiss
    return faiss


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer the question using ONLY the context passages provided.
If the context does not contain enough information, say: "I don't have enough information to answer that."
Do not add facts not present in the context. Be concise — 1-2 sentences."""


class RAGPipeline:
    """
    A minimal RAG pipeline: embed → index → retrieve → generate.

    Parameters
    ----------
    top_k : int
        Number of chunks to retrieve per query.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self._encoder = None
        self._index = None
        self._texts: list[str] = []
        self._ids: list[str] = []
        self._llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Index building ────────────────────────────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        """
        Embed all documents and build a FAISS index.

        Each document must have keys: id, text.
        Call this once at startup — it takes ~2 seconds for 25 docs.
        """
        faiss = _get_faiss()
        self._encoder = _get_encoder()

        self._texts = [d["text"] for d in documents]
        self._ids   = [d["id"]   for d in documents]

        embeddings = self._encoder.encode(self._texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalisation
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> tuple[list[str], list[float]]:
        """
        Find the top-k most relevant chunks for a question.

        Returns (chunks, scores). Scores are cosine similarities (0–1).
        """
        if self._index is None:
            raise RuntimeError("Call build_index() before retrieve()")

        faiss = _get_faiss()
        q_emb = self._encoder.encode([question], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self._index.search(q_emb, self.top_k)
        chunks = [self._texts[i] for i in indices[0]]
        sims   = [float(scores[0][j]) for j in range(len(indices[0]))]
        return chunks, sims

    # ── Generation ────────────────────────────────────────────────────────

    def generate(self, question: str, chunks: list[str]) -> str:
        """
        Ask Claude to answer the question using only the retrieved chunks.
        """
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
        r = self._llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return r.content[0].text.strip()

    # ── End-to-end ────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Retrieve + generate. Returns a dict with question, answer, and chunks.
        """
        chunks, scores = self.retrieve(question)
        answer = self.generate(question, chunks)
        return {
            "question": question,
            "answer":   answer,
            "contexts": chunks,
            "scores":   scores,
        }


# ── Manual smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from pipeline.corpus import DOCUMENTS

    console = Console()

    console.print(Rule("[bold]RAG Pipeline — smoke test[/bold]"))
    console.print("[dim]Building index...[/dim]")

    rag = RAGPipeline(top_k=3)
    rag.build_index(DOCUMENTS)

    console.print("[green]✓ Index ready[/green]\n")

    questions = [
        "Who directed Inception?",
        "How many Oscars did Oppenheimer win?",
    ]

    for q in questions:
        result = rag.ask(q)
        console.print(f"[bold cyan]Q:[/bold cyan] {result['question']}")
        for i, (chunk, score) in enumerate(zip(result["contexts"], result["scores"]), 1):
            console.print(f"  [dim]chunk {i} (score {score:.2f}):[/dim] {chunk[:80]}...")
        console.print(Panel(result["answer"], title="Answer", border_style="green"))
        console.print()
