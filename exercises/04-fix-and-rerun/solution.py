"""
Exercise 4 — Solution
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
import anthropic

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

from pipeline.corpus import DOCUMENTS, EVAL_QUESTIONS
from pipeline.rag import RAGPipeline

console = Console()

judge_llm   = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BROKEN_PROMPT = """You are a helpful assistant that answers questions about movies.
Answer the question as best you can."""


def _generate_with_prompt(rag, question, system_prompt):
    chunks, _ = rag.retrieve(question)
    llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    r = llm.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return r.content[0].text.strip(), chunks


def run_pipeline(top_k=3, system_prompt_override=None):
    rag = RAGPipeline(top_k=top_k)
    rag.build_index(DOCUMENTS)
    results = []
    for item in EVAL_QUESTIONS:
        if system_prompt_override:
            answer, contexts = _generate_with_prompt(rag, item["question"], system_prompt_override)
            results.append({"question": item["question"], "answer": answer, "contexts": contexts, "ground_truth": item["ground_truth"]})
        else:
            r = rag.ask(item["question"])
            r["ground_truth"] = item["ground_truth"]
            results.append(r)
    return results


def score(results):
    ds = Dataset.from_list([{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"], "ground_truth": r["ground_truth"]} for r in results])
    out = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall], llm=judge_llm, embeddings=embed_model, raise_exceptions=False)
    df = out.to_pandas()
    return {k: round(df[k].dropna().mean(), 3) for k in ["faithfulness", "answer_relevancy", "context_recall"]}


def _fmt(v):
    f = float(v)
    c = "green" if f >= 0.7 else ("yellow" if f >= 0.4 else "red")
    return f"[{c}]{f:.2f}[/{c}]"


console.print("[dim]Running baseline...[/dim]")
baseline = score(run_pipeline(top_k=3))

console.print("[dim]Running Break A (top_k=1)...[/dim]")
broken_a = score(run_pipeline(top_k=1))

console.print("[dim]Running Break B (no grounding)...[/dim]")
broken_b = score(run_pipeline(system_prompt_override=BROKEN_PROMPT))

table = Table(show_header=True, header_style="bold")
table.add_column("Pipeline version")
table.add_column("Faithfulness",     justify="right")
table.add_column("Answer Relevancy", justify="right")
table.add_column("Context Recall",   justify="right")

for label, s in [("Baseline (top_k=3)", baseline), ("Break A: top_k=1", broken_a), ("Break B: no grounding", broken_b)]:
    table.add_row(label, _fmt(s["faithfulness"]), _fmt(s["answer_relevancy"]), _fmt(s["context_recall"]))

console.print(table)
