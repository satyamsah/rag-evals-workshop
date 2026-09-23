"""
Exercise 3 — RAGAS Scores
============================
RAGAS automates what you just did by hand in Exercise 2.

It uses an LLM internally (Claude haiku) as a judge to score:
  - Faithfulness     — is the answer grounded in the retrieved context?
  - Answer Relevancy — does the answer address the question?
  - Context Recall   — did retrieval find the document containing the answer?

Your job: uncomment the block below and run it.
Compare the scores to what you wrote down in Exercise 2.

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

# ── Step 1: Run the pipeline (already done for you) ───────────────────────

console.print("[dim]Building index and running pipeline on 10 questions...[/dim]")

rag = RAGPipeline(top_k=3)
rag.build_index(DOCUMENTS)

results = []
for item in EVAL_QUESTIONS:
    r = rag.ask(item["question"])
    r["ground_truth"] = item["ground_truth"]
    results.append(r)

console.print(f"[green]✓ {len(results)} answers generated[/green]\n")


# ── Step 2: Uncomment this entire block to run RAGAS ─────────────────────
#
# RAGAS needs four things for each question:
#   user_input          — what was asked
#   response            — what the RAG system said
#   retrieved_contexts  — the retrieved chunks (as a list)
#   reference           — the correct answer
#
# It uses Claude haiku as an internal judge to score each answer.
# This takes about 60–90 seconds to run.
#
# TODO: UNCOMMENT THE ENTIRE BLOCK BELOW AND RE-RUN
#
# from ragas import evaluate, EvaluationDataset, SingleTurnSample
# from ragas.metrics import faithfulness, answer_relevancy, context_recall
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_anthropic import ChatAnthropic
# from langchain_community.embeddings import HuggingFaceEmbeddings
#
# dataset = EvaluationDataset(samples=[
#     SingleTurnSample(
#         user_input=r["question"],
#         response=r["answer"],
#         retrieved_contexts=r["contexts"],
#         reference=r["ground_truth"],
#     )
#     for r in results
# ])
#
# judge_llm   = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY")))
# embed_model = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
#
# console.print("[dim]Running RAGAS evaluation — this takes ~60–90 seconds...[/dim]\n")
# scores = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_recall],
#     llm=judge_llm,
#     embeddings=embed_model,
#     raise_exceptions=False,
# )
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
#         row["user_input"][:44],
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
# console.print("[dim]Compare these scores to what you wrote in Exercise 2. Where do they disagree?[/dim]")

console.print("[dim]Uncomment the block above (Step 2) and re-run to see the RAGAS scores.[/dim]")
