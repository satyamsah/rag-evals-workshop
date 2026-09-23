"""
Exercise 2 — Grade by Hand
============================
The pipeline runs for you. Your job is to score the answers.

For each question you will see:
  - The chunks that were retrieved
  - The answer the RAG system gave
  - The correct answer (ground truth)

Score each answer on these three things:
  Faithfulness     — is the answer supported by the retrieved chunks?  (0 = no, 1 = yes)
  Answer Relevancy — does it actually answer the question?             (0 = no, 1 = yes)
  Context Recall   — did retrieval surface the right document?         (0 = no, 1 = yes)

Then fill in your scores in the SCORECARD at the bottom.
In Exercise 3 RAGAS will score the same answers automatically — compare the two.

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

console.print("[green]✓ Done — read through each answer below and fill in your scores\n[/green]")

# ── Display each question ─────────────────────────────────────────────────

for i, result in enumerate(results, 1):
    console.print(Rule(f"[bold]Question {i} of {len(results)}[/bold]"))
    console.print(f"[bold cyan]Q:[/bold cyan] {result['question']}\n")

    console.print("[dim]Retrieved chunks (what the LLM was given):[/dim]")
    for j, (chunk, score) in enumerate(zip(result["contexts"], result["scores"]), 1):
        console.print(Panel(chunk, title=f"[dim]Chunk {j}  (similarity={score:.2f})[/dim]", border_style="dim"))

    console.print(Panel(result["answer"],       title="[green]RAG Answer[/green]",     border_style="green"))
    console.print(Panel(result["ground_truth"], title="[yellow]Correct Answer[/yellow]", border_style="yellow"))
    console.print()

# ── Your scorecard ────────────────────────────────────────────────────────

console.print(Rule("[bold]Your Scorecard[/bold]"))
console.print("""
Fill in your scores below (0 = no, 1 = yes) then save the file.
In Exercise 3 you will see how RAGAS scores compare.

Ask yourself for each answer:
  Faithfulness     — does the answer match what was in the retrieved chunks?
  Answer Relevancy — does it directly answer what was asked?
  Context Recall   — was the right document in the retrieved chunks?
""")

# ── Uncomment this block and fill in your scores ──────────────────────────

# SCORECARD = [
#     {"q": "Who directed Inception?",
#      "faithfulness": ?,  "relevancy": ?,  "recall": ?},
#
#     {"q": "Which actor won an award for playing the Joker?",
#      "faithfulness": ?,  "relevancy": ?,  "recall": ?},
#
#     {"q": "What was the first non-English film to win Best Picture?",
#      "faithfulness": ?,  "relevancy": ?,  "recall": ?},
#
#     {"q": "Who plays Evelyn in Everything Everywhere All at Once?",
#      "faithfulness": ?,  "relevancy": ?,  "recall": ?},
#
#     {"q": "How many Oscars did Oppenheimer win?",
#      "faithfulness": ?,  "relevancy": ?,  "recall": ?},
# ]
#
# for row in SCORECARD:
#     print(f"  {row['q'][:50]:<50}  faith={row['faithfulness']}  rel={row['relevancy']}  recall={row['recall']}")
