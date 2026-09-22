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

> **During the session — open [`WORKSHOP.md`](WORKSHOP.md) and follow it top to bottom.**

---

## Schedule

| Time | Block | Format |
|------|-------|--------|
| 0:00–0:15 | Setup & welcome | Setup |
| 0:15–0:40 | What is RAG and why evaluate it? | Talk + demo |
| 0:40–1:00 | Exercise 1 — Build the pipeline | Hands-on |
| 1:00–1:15 | ☕ Break | — |
| 1:15–1:35 | What makes a good RAG answer? | Talk |
| 1:35–2:00 | Exercise 2 — Grade by hand | Hands-on |
| 2:00–2:15 | RAGAS — automate the grading | Talk + demo |
| 2:15–2:30 | Exercise 3 — RAGAS scores | Hands-on |
| 2:30–2:40 | ☕ Short break | — |
| 2:40–3:00 | Exercise 4 — Break it, fix it | Hands-on |
| 3:00–3:10 | What to do when scores are bad | Talk |
| 3:10–3:30 | End-to-end review + Q&A | Live demo |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | `python3 --version` to check |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) — free tier works |
| A terminal | Terminal on Mac, PowerShell or Command Prompt on Windows |

No prior eval or RAG experience needed.

---

## Setup (do this before the session)

### 1. Get the code

```bash
git clone https://github.com/satyamsah/rag-evals-workshop
cd rag-evals-workshop
```

### 2. Set up Python

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first `pip install` takes 2–3 minutes — `sentence-transformers` is a large package.

### 3. Add your API key

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx
```

**Where to get a key:**
1. Go to https://console.anthropic.com
2. Sign up (free) or log in
3. Click **Get API Keys** → **Create Key**
4. Copy and paste into `.env`

> The `.env` file is in `.gitignore` — it will never be pushed to GitHub.

### 4. Verify everything works

```bash
python check_setup.py
```

You should see:
```
✓ Python 3.x
✓ python-dotenv installed
✓ ANTHROPIC_API_KEY found
✓ Anthropic API reachable
✓ ragas installed
✓ datasets installed
✓ faiss-cpu installed
✓ Ready for the workshop!
```

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
