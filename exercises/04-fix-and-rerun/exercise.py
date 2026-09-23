"""
Exercise 4 — Break It, Fix It
================================
The best way to understand an eval metric is to watch it move.

You will make two deliberate changes to the pipeline and observe which
metric drops each time. Then you will understand exactly what each metric
is measuring.

Break A — Retrieval:  change top_k from 3 to 1
  → only one chunk retrieved per question
  → which metric drops?

Break B — Prompt:     remove the grounding instruction
  → LLM is no longer told to stay within the retrieved docs
  → which metric drops this time?

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
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
import anthropic

load_dotenv()

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()

judge_llm   = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Helper: score a list of results with RAGAS ────────────────────────────

def score(results):
    ds = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"],
         "contexts": r["contexts"], "ground_truth": r["ground_truth"]}
        for r in results
    ])
    out = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall],
                   llm=judge_llm, embeddings=embed_model, raise_exceptions=False)
    df = out.to_pandas()
    return {k: round(df[k].dropna().mean(), 3) for k in ["faithfulness", "answer_relevancy", "context_recall"]}


def run_pipeline(top_k=3, system_prompt=None):
    rag = RAGPipeline(top_k=top_k)
    rag.build_index(DOCUMENTS)
    results = []
    for item in EVAL_QUESTIONS:
        if system_prompt:
            chunks, _ = rag.retrieve(item["question"])
            context   = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
            llm       = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            r         = llm.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=200,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {item['question']}"}],
            )
            results.append({"question": item["question"], "answer": r.content[0].text.strip(),
                             "contexts": chunks, "ground_truth": item["ground_truth"]})
        else:
            r = rag.ask(item["question"])
            r["ground_truth"] = item["ground_truth"]
            results.append(r)
    return results


def _fmt(v):
    f = float(v)
    c = "green" if f >= 0.7 else ("yellow" if f >= 0.4 else "red")
    return f"[{c}]{f:.2f}[/{c}]"


# ── Baseline ──────────────────────────────────────────────────────────────

console.print(Rule("[bold]Baseline — top_k=3, grounding prompt on[/bold]"))
console.print("[dim]Running...[/dim]")

# Step 1 — uncomment to run the baseline
# baseline_results = run_pipeline(top_k=3)
# baseline_scores  = score(baseline_results)
# console.print(f"Faithfulness:     {_fmt(baseline_scores['faithfulness'])}")
# console.print(f"Answer Relevancy: {_fmt(baseline_scores['answer_relevancy'])}")
# console.print(f"Context Recall:   {_fmt(baseline_scores['context_recall'])}\n")


# ── Break A: top_k=1 ──────────────────────────────────────────────────────

console.print(Rule("[bold]Break A — top_k=1 (retrieval gets worse)[/bold]"))
console.print("[dim]Only 1 chunk retrieved instead of 3. Which metric drops?[/dim]\n")

# Step 2 — uncomment to run Break A
# broken_a_results = run_pipeline(top_k=1)
# broken_a_scores  = score(broken_a_results)
# console.print(f"Faithfulness:     {_fmt(broken_a_scores['faithfulness'])}")
# console.print(f"Answer Relevancy: {_fmt(broken_a_scores['answer_relevancy'])}")
# console.print(f"Context Recall:   {_fmt(broken_a_scores['context_recall'])}\n")


# ── Break B: no grounding instruction ────────────────────────────────────

BROKEN_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer the question as best you can."""
# ↑ grounding instruction removed — LLM can now use its training memory

console.print(Rule("[bold]Break B — no grounding instruction (LLM goes off-script)[/bold]"))
console.print("[dim]System prompt no longer says 'answer ONLY from context'. Which metric drops?[/dim]\n")

# Step 3 — uncomment to run Break B
# broken_b_results = run_pipeline(system_prompt=BROKEN_PROMPT)
# broken_b_scores  = score(broken_b_results)
# console.print(f"Faithfulness:     {_fmt(broken_b_scores['faithfulness'])}")
# console.print(f"Answer Relevancy: {_fmt(broken_b_scores['answer_relevancy'])}")
# console.print(f"Context Recall:   {_fmt(broken_b_scores['context_recall'])}\n")


# ── Summary table ─────────────────────────────────────────────────────────

# Step 4 — uncomment to see the comparison table
# console.print(Rule("[bold]Summary[/bold]"))
# table = Table(show_header=True, header_style="bold")
# table.add_column("Pipeline version")
# table.add_column("Faithfulness",     justify="right")
# table.add_column("Answer Relevancy", justify="right")
# table.add_column("Context Recall",   justify="right")
# table.add_row("Baseline (top_k=3)",     _fmt(baseline_scores["faithfulness"]), _fmt(baseline_scores["answer_relevancy"]), _fmt(baseline_scores["context_recall"]))
# table.add_row("Break A: top_k=1",       _fmt(broken_a_scores["faithfulness"]), _fmt(broken_a_scores["answer_relevancy"]), _fmt(broken_a_scores["context_recall"]))
# table.add_row("Break B: no grounding",  _fmt(broken_b_scores["faithfulness"]), _fmt(broken_b_scores["answer_relevancy"]), _fmt(broken_b_scores["context_recall"]))
# console.print(table)
# console.print()
# console.print("[dim]Break A hurts Context Recall. Break B hurts Faithfulness. Different problems → different metrics.[/dim]")

console.print("[dim]Uncomment the steps above one at a time and re-run.[/dim]")
