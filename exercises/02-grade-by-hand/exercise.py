"""
Exercise 2 — Grade by Hand
============================
Before trusting a number, understand what it measures.

In this exercise you will manually score 5 RAG answers against three criteria:
  - Faithfulness:      Is the answer supported by the retrieved context?
  - Answer Relevancy:  Does the answer actually address the question?
  - Context Recall:    Did retrieval surface the document that contains the answer?

You will fill in a scorecard, then compare your scores to RAGAS in Exercise 3.
The goal: build intuition so you know what to do when a metric drops.

Run:  python exercises/02-grade-by-hand/exercise.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()

# ── Build the pipeline ────────────────────────────────────────────────────
console.print("[dim]Building index...[/dim]")
rag = RAGPipeline(top_k=3)
rag.build_index(DOCUMENTS)
console.print("[green]✓ Ready[/green]\n")

# ── Pick 5 questions to grade ─────────────────────────────────────────────
SAMPLE = EVAL_QUESTIONS[:5]

# ── Scoring guide ─────────────────────────────────────────────────────────
console.print(Panel(
    "[bold]Scoring guide[/bold]\n\n"
    "For each question you will see:\n"
    "  • The retrieved chunks  (what the RAG system found)\n"
    "  • The generated answer  (what the LLM said)\n"
    "  • The ground truth      (the correct answer)\n\n"
    "Score each dimension 0–3:\n"
    "  [bold]Faithfulness[/bold]:      0 = answer contradicts context  |  3 = fully grounded\n"
    "  [bold]Answer Relevancy[/bold]:  0 = off-topic answer            |  3 = directly answers question\n"
    "  [bold]Context Recall[/bold]:    0 = wrong docs retrieved        |  3 = right doc in top-3\n\n"
    "Write your scores in the SCORECARD at the bottom of this file.",
    border_style="cyan",
))
console.print()

# ── Run and display ───────────────────────────────────────────────────────
results = []

for i, item in enumerate(SAMPLE, 1):
    q  = item["question"]
    gt = item["ground_truth"]

    result = rag.ask(q)
    result["ground_truth"] = gt
    results.append(result)

    console.print(Rule(f"[bold]Question {i} of {len(SAMPLE)}[/bold]"))
    console.print(f"[bold cyan]Q:[/bold cyan] {q}\n")

    console.print("[dim]Retrieved chunks:[/dim]")
    for j, (chunk, score) in enumerate(zip(result["contexts"], result["scores"]), 1):
        console.print(Panel(chunk, title=f"[dim]Chunk {j}  (sim={score:.2f})[/dim]", border_style="dim"))

    console.print(Panel(result["answer"],   title="[green]RAG Answer[/green]",    border_style="green"))
    console.print(Panel(gt,                 title="[yellow]Ground Truth[/yellow]", border_style="yellow"))
    console.print()


# ── YOUR SCORECARD ─────────────────────────────────────────────────────────
#
# Fill in your scores below. 0 = bad, 3 = perfect.
#
# Then run Exercise 3 to see how RAGAS scores compare.
#
# SCORECARD = [
#     # question                                          faith  rel  recall
#     {"q": "Who directed Inception?",                    "faithfulness": ?, "relevancy": ?, "recall": ?},
#     {"q": "Which actor won an award for the Joker?",   "faithfulness": ?, "relevancy": ?, "recall": ?},
#     {"q": "First non-English Best Picture?",           "faithfulness": ?, "relevancy": ?, "recall": ?},
#     {"q": "Who plays Evelyn in EEAAO?",                "faithfulness": ?, "relevancy": ?, "recall": ?},
#     {"q": "How many Oscars did Oppenheimer win?",      "faithfulness": ?, "relevancy": ?, "recall": ?},
# ]
#
console.print(Panel(
    "[bold]Your turn[/bold]\n\n"
    "Fill in SCORECARD above for each question, then move to Exercise 3.\n\n"
    "Key question to keep in mind:\n"
    "  When the answer was wrong — was the [yellow]retrieved chunk[/yellow] wrong, "
    "or did the [green]LLM[/green] go off-script?\n"
    "  That distinction tells you whether to fix retrieval or the prompt.",
    border_style="bold",
))
