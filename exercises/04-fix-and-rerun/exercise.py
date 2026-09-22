"""
Exercise 4 — Break It, Fix It
================================
The best way to understand an eval metric is to watch it move.

In this exercise you will:
  1. Deliberately break the pipeline two ways
  2. Run RAGAS and observe which scores drop
  3. Apply a fix and watch the scores recover

Breaks we will make:
  A. Retrieval break — reduce top_k from 3 to 1
     → context_recall will drop because the right document is often not in top-1
  B. Prompt break    — remove the grounding instruction from the system prompt
     → faithfulness will drop because the LLM starts adding facts from memory

Run:  python exercises/04-fix-and-rerun/exercise.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()


# ── Helper: run RAGAS on a list of result dicts ───────────────────────────

judge_llm   = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def score(results: list[dict]) -> dict:
    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ])
    out = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=judge_llm,
        embeddings=embed_model,
        raise_exceptions=False,
    )
    df = out.to_pandas()
    return {
        "faithfulness":     round(df["faithfulness"].dropna().mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].dropna().mean(), 3),
        "context_recall":   round(df["context_recall"].dropna().mean(), 3),
    }


def run_pipeline(top_k: int = 3, system_prompt_override: str = None) -> list[dict]:
    rag = RAGPipeline(top_k=top_k)
    if system_prompt_override is not None:
        rag._system_prompt = system_prompt_override  # monkey-patch for demo
    rag.build_index(DOCUMENTS)

    results = []
    for item in EVAL_QUESTIONS:
        r = rag.ask(item["question"])
        r["ground_truth"] = item["ground_truth"]
        # apply prompt override at generation time
        if system_prompt_override is not None:
            chunks, _ = rag.retrieve(item["question"])
            r["answer"] = _generate_with_prompt(item["question"], chunks, system_prompt_override)
        results.append(r)
    return results


def _generate_with_prompt(question, chunks, system_prompt):
    import anthropic, os
    llm     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    r = llm.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return r.content[0].text.strip()


# ── Baseline ──────────────────────────────────────────────────────────────

console.print(Rule("[bold]Step 1 — Baseline scores[/bold]"))
console.print("[dim]Running baseline pipeline (top_k=3, correct prompt)...[/dim]")
baseline_results = run_pipeline(top_k=3)
baseline_scores  = score(baseline_results)
console.print(f"[green]✓ Baseline done[/green]: {baseline_scores}\n")


# ── Break A: Retrieval (top_k=1) ──────────────────────────────────────────

console.print(Rule("[bold]Step 2 — Break A: top_k=1[/bold]"))
console.print("[dim]Now only retrieving 1 chunk instead of 3...[/dim]")

# TODO: run run_pipeline(top_k=1) and call score()
# broken_a_scores = score(run_pipeline(top_k=1))
broken_a_scores = {"faithfulness": "TODO", "answer_relevancy": "TODO", "context_recall": "TODO"}

console.print(f"Broken (top_k=1): {broken_a_scores}\n")
console.print("[dim]Which metric dropped? Why? How much does context_recall change?[/dim]\n")


# ── Break B: Prompt (no grounding instruction) ────────────────────────────

BROKEN_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer the question as best you can."""  # grounding instruction removed

console.print(Rule("[bold]Step 3 — Break B: no grounding instruction[/bold]"))
console.print("[dim]Now running with a prompt that doesn't say 'answer ONLY from context'...[/dim]")

# TODO: run run_pipeline(system_prompt_override=BROKEN_PROMPT) and call score()
# broken_b_scores = score(run_pipeline(system_prompt_override=BROKEN_PROMPT))
broken_b_scores = {"faithfulness": "TODO", "answer_relevancy": "TODO", "context_recall": "TODO"}

console.print(f"Broken (no grounding): {broken_b_scores}\n")
console.print("[dim]Which metric dropped this time? Is it the same one as Break A?[/dim]\n")


# ── Summary table ─────────────────────────────────────────────────────────

console.print(Rule("[bold]Summary — what the metrics caught[/bold]"))

table = Table(show_header=True, header_style="bold")
table.add_column("Pipeline version")
table.add_column("Faithfulness",     justify="right")
table.add_column("Answer Relevancy", justify="right")
table.add_column("Context Recall",   justify="right")

def _fmt(v):
    if v == "TODO":
        return "[dim]TODO[/dim]"
    f = float(v)
    c = "green" if f >= 0.7 else ("yellow" if f >= 0.4 else "red")
    return f"[{c}]{f:.2f}[/{c}]"

table.add_row("Baseline (top_k=3)",      _fmt(baseline_scores["faithfulness"]), _fmt(baseline_scores["answer_relevancy"]), _fmt(baseline_scores["context_recall"]))
table.add_row("Break A: top_k=1",        _fmt(broken_a_scores["faithfulness"]), _fmt(broken_a_scores["answer_relevancy"]), _fmt(broken_a_scores["context_recall"]))
table.add_row("Break B: no grounding",   _fmt(broken_b_scores["faithfulness"]), _fmt(broken_b_scores["answer_relevancy"]), _fmt(broken_b_scores["context_recall"]))

console.print(table)
console.print()
console.print("[dim]Stuck? See exercises/04-fix-and-rerun/solution.py[/dim]")
