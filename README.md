# RAG Evals with RAGAS
### A 2.5-hour hands-on workshop

> Build a RAG pipeline, score it with RAGAS, break it deliberately, and fix it.

---

## What you will build

A movie-review RAG pipeline that you can actually measure:

```
Q: Who directed Inception?
A: Inception was directed by Christopher Nolan.

Faithfulness:     0.97   ← answer is grounded in retrieved context
Answer Relevancy: 0.91   ← answer addresses the question
Context Recall:   0.90   ← the right document was retrieved
```

Then you will deliberately break it and watch the scores fall — so you understand exactly what each metric catches.

---

## During the session

Open **[WORKSHOP.md](WORKSHOP.md)** and follow it top to bottom. Everything — setup, exercises, explanations — is in there.

---

## Directory layout

```
rag-evals-workshop/
├── WORKSHOP.md               ← follow this during the session
├── README.md                 ← you are here
├── requirements.txt
├── check_setup.py
├── .env.example
│
├── pipeline/                 the RAG pipeline (complete reference version)
│   ├── corpus.py             document store + eval questions
│   └── rag.py                embed → index → retrieve → generate
│
└── exercises/
    ├── 01-build-rag/         Exercise 1 — assemble the pipeline
    ├── 02-grade-by-hand/     Exercise 2 — manual scoring
    ├── 03-ragas-scores/      Exercise 3 — automated RAGAS eval
    └── 04-fix-and-rerun/     Exercise 4 — break it and fix it
```
