"""
Exercise 2 — Grade by Hand
============================
The pipeline runs automatically. Your job is to read each answer and score it.

For each of the 5 questions you will see:
  - The chunks the RAG system retrieved
  - The answer the RAG system gave
  - The correct answer (ground truth)

Then the terminal will ask you to score three things (type 0 or 1):
  Faithfulness     — is the answer supported by the retrieved chunks?
  Answer Relevancy — does it actually answer the question?
  Context Recall   — was the right information in the retrieved chunks?

At the end you will see your scorecard.
In Exercise 3, RAGAS will score the same answers automatically — compare the two.

Run:  python exercises/02-grade-by-hand/exercise.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

load_dotenv()

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()

console.print("[dim]Building index and running pipeline on 5 questions...[/dim]")
rag = RAGPipeline(top_k=3)
rag.build_index(DOCUMENTS)

SAMPLE = EVAL_QUESTIONS[:5]
results = []

for item in SAMPLE:
    r = rag.ask(item["question"])
    r["ground_truth"] = item["ground_truth"]
    results.append(r)

console.print("[green]✓ Done — read each answer carefully and score it\n[/green]")


def ask_score(label):
    """Ask the user to enter 0 or 1. Keeps asking until valid."""
    while True:
        val = input(f"    {label} (0 = no / 1 = yes): ").strip()
        if val in ("0", "1"):
            return int(val)
        console.print("    [red]Please enter 0 or 1[/red]")


# ── Go through each question ──────────────────────────────────────────────

scorecard = []

for i, result in enumerate(results, 1):
    console.print(Rule(f"[bold]Question {i} of {len(results)}[/bold]"))
    console.print(f"[bold cyan]Q:[/bold cyan] {result['question']}\n")

    console.print("[dim]Retrieved chunks (what the LLM was given):[/dim]")
    for j, (chunk, score) in enumerate(zip(result["contexts"], result["scores"]), 1):
        console.print(Panel(chunk, title=f"[dim]Chunk {j}  (similarity={score:.2f})[/dim]", border_style="dim"))

    console.print(Panel(result["answer"],       title="[green]RAG Answer[/green]",      border_style="green"))
    console.print(Panel(result["ground_truth"], title="[yellow]Correct Answer[/yellow]", border_style="yellow"))

    console.print("\n[bold]Your scores:[/bold]")
    console.print("  [dim]Faithfulness     — does the RAG answer match the retrieved chunks?[/dim]")
    console.print("  [dim]Answer Relevancy — does it directly answer the question?[/dim]")
    console.print("  [dim]Context Recall   — was the right information in the chunks?[/dim]\n")

    faith   = ask_score("Faithfulness    ")
    rel     = ask_score("Answer Relevancy")
    recall  = ask_score("Context Recall  ")

    scorecard.append({
        "question": result["question"],
        "faithfulness": faith,
        "relevancy": rel,
        "recall": recall,
    })
    console.print()


# ── Summary table ─────────────────────────────────────────────────────────

console.print(Rule("[bold]Your Scorecard[/bold]"))

table = Table(show_header=True, header_style="bold")
table.add_column("Question", style="cyan", max_width=45)
table.add_column("Faithfulness", justify="center")
table.add_column("Answer Relevancy", justify="center")
table.add_column("Context Recall", justify="center")

for row in scorecard:
    def fmt(v):
        return "[green]1[/green]" if v == 1 else "[red]0[/red]"
    table.add_row(row["question"][:45], fmt(row["faithfulness"]), fmt(row["relevancy"]), fmt(row["recall"]))

console.print(table)
console.print("\n[dim]Save these scores — in Exercise 3 you will compare them to RAGAS.[/dim]\n")
