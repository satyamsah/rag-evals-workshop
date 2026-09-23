"""
Exercise 3 — RAGAS Scores
============================
RAGAS automates what you just did by hand in Exercise 2.

It uses an LLM internally (Claude haiku) as a judge to score:
  - Faithfulness     — is the answer grounded in the retrieved context?
  - Answer Relevancy — does the answer address the question?
  - Context Recall   — did retrieval find the document containing the answer?

Why do we need LangChain here?
  RAGAS was built to work with LangChain's model interface.
  We point it at Claude haiku — same model as before, just wrapped differently.
  Your pipeline code (anthropic SDK) stays unchanged.

Run:  python exercises/03-ragas-scores/exercise.py
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

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()

# ── Step 1: Run the pipeline on all 10 eval questions ─────────────────────

console.print("[dim]Building index and running pipeline on 10 questions...[/dim]")

rag = RAGPipeline(top_k=3)
rag.build_index(DOCUMENTS)

results = []
for item in EVAL_QUESTIONS:
    r = rag.ask(item["question"])
    r["ground_truth"] = item["ground_truth"]
    results.append(r)

console.print(f"[green]✓ {len(results)} answers generated[/green]\n")


# ── Step 2: Format for RAGAS ──────────────────────────────────────────────
#
# RAGAS expects four columns:
#   question      — what was asked
#   answer        — what the RAG system said
#   contexts      — the list of retrieved chunks
#   ground_truth  — the correct answer (used to measure context recall)
#
# Uncomment each line below one at a time and re-run.
#

# Step 2a — import the RAGAS libraries
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall
# from langchain_anthropic import ChatAnthropic
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 2b — build the dataset RAGAS expects
# dataset = Dataset.from_list([
#     {
#         "question":     r["question"],
#         "answer":       r["answer"],
#         "contexts":     r["contexts"],
#         "ground_truth": r["ground_truth"],
#     }
#     for r in results
# ])

# Step 2c — set up the LLM judge and embedding model
# judge_llm   = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))
# embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2d — run the evaluation (this takes ~60–90 seconds)
# console.print("[dim]Running RAGAS evaluation...[/dim]\n")
# scores = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_recall],
#     llm=judge_llm,
#     embeddings=embed_model,
#     raise_exceptions=False,
# )


# ── Step 3: Display results ───────────────────────────────────────────────
#
# Uncomment once Step 2 is complete.
#

# df = scores.to_pandas()
#
# console.print(Rule("[bold]RAGAS Scores — per question[/bold]"))
#
# table = Table(show_header=True, header_style="bold")
# table.add_column("Question",         style="cyan", max_width=45)
# table.add_column("Faithfulness",     justify="right")
# table.add_column("Answer Relevancy", justify="right")
# table.add_column("Context Recall",   justify="right")
#
# def _fmt(val):
#     if val is None or str(val) == "nan":
#         return "[dim]—[/dim]"
#     f = float(val)
#     colour = "green" if f >= 0.7 else ("yellow" if f >= 0.4 else "red")
#     return f"[{colour}]{f:.2f}[/{colour}]"
#
# for _, row in df.iterrows():
#     table.add_row(
#         row["question"][:44],
#         _fmt(row.get("faithfulness")),
#         _fmt(row.get("answer_relevancy")),
#         _fmt(row.get("context_recall")),
#     )
# console.print(table)
# console.print()
#
# console.print(Rule("[bold]Aggregate[/bold]"))
# for metric, label in [("faithfulness","Faithfulness"), ("answer_relevancy","Answer Relevancy"), ("context_recall","Context Recall")]:
#     mean = df[metric].dropna().mean() if metric in df else None
#     console.print(f"  {label:<20} {_fmt(mean)}")
# console.print()
# console.print("[dim]Scores below 0.7 are worth investigating. Move to Exercise 4 to break and fix the pipeline.[/dim]")

console.print("[dim]Uncomment the steps above one at a time and re-run to see the scores.[/dim]")
