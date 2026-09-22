"""
Exercise 3 — RAGAS Scores
============================
Now let RAGAS do what you just did by hand — at scale, automatically.

RAGAS runs an LLM-as-judge internally to score each answer against:
  - faithfulness:       Is the answer grounded in the retrieved context?
  - answer_relevancy:   Does the answer address the question?
  - context_recall:     Did retrieval surface the document containing the answer?
                        (requires a ground_truth reference)

We will score all 10 questions and print a summary table.

Run:  python exercises/03-ragas-scores/exercise.py

── What to look for ─────────────────────────────────────────────────────────

  Any score below 0.7 is a problem. Note WHICH questions score low and WHY.
  • Low faithfulness  → LLM is adding facts not in the retrieved chunks
  • Low relevancy     → LLM is answering a slightly different question
  • Low context_recall → the right document was not retrieved at all

  You will deliberately break the pipeline in Exercise 4 and watch scores drop.
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

# ── Step 1: Run the pipeline on all eval questions ────────────────────────

console.print("[dim]Building index and running pipeline on 10 questions...[/dim]")

rag = RAGPipeline(top_k=3)
rag.build_index(DOCUMENTS)

results = []
for item in EVAL_QUESTIONS:
    r = rag.ask(item["question"])
    r["ground_truth"] = item["ground_truth"]
    results.append(r)

console.print(f"[green]✓ {len(results)} answers generated[/green]\n")


# ── Step 2: Format for RAGAS ───────────────────────────────────────────────
#
# RAGAS expects a HuggingFace Dataset with columns:
#   question, answer, contexts (list of strings), ground_truth
#

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

console.print("[dim]Running RAGAS evaluation (this takes ~60–90 seconds)...[/dim]\n")

dataset = Dataset.from_list([
    {
        "question":     r["question"],
        "answer":       r["answer"],
        "contexts":     r["contexts"],
        "ground_truth": r["ground_truth"],
    }
    for r in results
])

# RAGAS uses an LLM internally to judge faithfulness and relevancy.
# We point it at Claude haiku to keep costs low.
judge_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=judge_llm,
    embeddings=embed_model,
    raise_exceptions=False,
)

# ── Step 3: Display results ────────────────────────────────────────────────

console.print(Rule("[bold]RAGAS Scores — per question[/bold]"))

df = scores.to_pandas()

table = Table(show_header=True, header_style="bold")
table.add_column("Question",          style="cyan",  max_width=45)
table.add_column("Faithfulness",      justify="right")
table.add_column("Answer Relevancy",  justify="right")
table.add_column("Context Recall",    justify="right")

def _fmt(val):
    if val is None or str(val) == "nan":
        return "[dim]—[/dim]"
    f = float(val)
    colour = "green" if f >= 0.7 else ("yellow" if f >= 0.4 else "red")
    return f"[{colour}]{f:.2f}[/{colour}]"

for _, row in df.iterrows():
    table.add_row(
        row["question"][:44],
        _fmt(row.get("faithfulness")),
        _fmt(row.get("answer_relevancy")),
        _fmt(row.get("context_recall")),
    )

console.print(table)
console.print()

# ── Aggregate ─────────────────────────────────────────────────────────────

console.print(Rule("[bold]Aggregate[/bold]"))
agg_table = Table(show_header=True, header_style="bold")
agg_table.add_column("Metric")
agg_table.add_column("Mean score", justify="right")
agg_table.add_column("Interpretation")

for metric, label, interpretation in [
    ("faithfulness",     "Faithfulness",     "Are answers grounded in retrieved context?"),
    ("answer_relevancy", "Answer Relevancy", "Do answers address the question?"),
    ("context_recall",   "Context Recall",   "Did retrieval find the right documents?"),
]:
    mean = df[metric].dropna().mean() if metric in df else None
    agg_table.add_row(label, _fmt(mean), interpretation)

console.print(agg_table)
console.print()
console.print("[dim]Scores < 0.7 are worth investigating. Move to Exercise 4 to break and fix the pipeline.[/dim]")
