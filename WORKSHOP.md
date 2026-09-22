# RAG Evals with RAGAS: Build, Score, and Fix Your Pipeline
### Workshop guide — follow this top to bottom during the session

---

## Schedule at a glance

| Time | Block |
|------|-------|
| 0:00 | Setup & welcome |
| 0:15 | Part 1 — What is RAG and why evaluate it? |
| 0:40 | Exercise 1 — Build the pipeline |
| 1:00 | ☕ Break |
| 1:15 | Part 2 — What makes a good RAG answer? |
| 1:35 | Exercise 2 — Grade by hand |
| 2:00 | Part 3 — RAGAS |
| 2:15 | Exercise 3 — RAGAS scores |
| 2:30 | ☕ Short break |
| 2:40 | Exercise 4 — Break it, fix it |
| 3:00 | Part 4 — When scores are bad |
| 3:10 | End-to-end review + Q&A |

---

## 0:00 — Setup & welcome

Ask everyone to run:

```bash
python check_setup.py
```

Expected output:
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

If anyone is stuck on setup, pair them with a neighbour — do not let a setup
issue derail the room. Move on after 10 minutes regardless.

---

## 0:15 — Part 1: What is RAG and why evaluate it?

### The problem RAG solves

LLMs are trained on data with a cutoff date. They cannot know:
- What your company's documents say
- What happened last week
- What is in your private database

RAG fixes this: retrieve relevant documents → pass them to the LLM → the LLM answers from that context instead of guessing from memory.

### The problem with RAG

RAG feels like it works. You try a few questions, the answers look reasonable, and you ship it.

Then, in production:
- Users report wrong answers
- You have no idea when it started
- You cannot tell if a change made it better or worse

**You are flying blind.** That is what this workshop is about.

### What "evaluating RAG" actually means

You need to measure three things:

```
Question  ──► Retrieval ──► Context ──► LLM ──► Answer
                 ↑                         ↑
         Did we find the             Did the LLM stay
         right documents?            within the documents?
```

| Metric | Question it answers | What breaks it |
|--------|---------------------|----------------|
| **Context Recall** | Did retrieval surface the document containing the answer? | top_k too small, vocabulary mismatch, bad embeddings |
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM ignores context and uses training memory |
| **Answer Relevancy** | Does the answer actually address the question? | Vague or off-topic generation |

### Why not just read the answers?

For 5 questions: fine. For 500 questions, or when you change the prompt and want to know if it got better — you need automation.

That is RAGAS: it runs an LLM-as-judge internally to score each answer on all three metrics.

---

## 0:40 — Exercise 1: Build the pipeline

**Goal:** assemble retrieve → generate from scratch so you understand each moving part.

```bash
python exercises/01-build-rag/exercise.py
```

**What the file does:**

`pipeline/corpus.py` — 25 movie-review passages (the document store) + 10 eval questions with ground-truth answers.

`exercises/01-build-rag/exercise.py` — three skeleton functions for you to fill in:

1. `build_index(documents)` — encode texts with SentenceTransformer, store in FAISS
2. `retrieve(question, index, texts)` — embed the question, find top-k nearest vectors
3. `generate(question, chunks)` — call Claude with the chunks as context

**How to run:**
```bash
python exercises/01-build-rag/exercise.py
```

**What you should see once complete:**
```
✓ Index built: 25 vectors

Q: Who directed Inception?
  chunk 1 (score 0.72): Inception is a 2010 sci-fi thriller directed by Christopher Nolan...
  chunk 2 (score 0.61): ...
Answer: Inception was directed by Christopher Nolan.
```

**Stuck?** See `exercises/01-build-rag/solution.py`.

> **Presenter note:** Walk through `build_index` live before releasing the room.
> Show what an embedding looks like: `encoder.encode(["hello world"])` → a 384-dim float array.
> The key insight: similar sentences have similar vectors — that is what makes retrieval work.

---

## 1:00 — ☕ Break

---

## 1:15 — Part 2: What makes a good RAG answer?

Before we automate scoring, we need to agree on what "good" means.

### Three things that can go wrong, independently

**Problem 1 — Retrieval retrieved the wrong document**

```
Q: Who plays Evelyn in Everything Everywhere All at Once?

Retrieved: "Everything Everywhere All at Once won seven Academy Awards..."
            ← this is about awards, not the cast

Answer: "I don't have enough information to answer that."
```

The answer is technically honest — but the problem is in retrieval, not generation.
`context_recall` will be low. Fix: increase `top_k`, improve the embedding model, or add metadata filters.

**Problem 2 — LLM ignored the context**

```
Q: How many Oscars did Oppenheimer win?

Retrieved: "Oppenheimer won seven Academy Awards in 2024..."  ← correct document retrieved

Answer: "Oppenheimer won 13 Academy Awards."  ← LLM used its training memory
```

The document was there. The LLM ignored it.
`faithfulness` will be low. Fix: strengthen the system prompt grounding instruction.

**Problem 3 — Answer doesn't address the question**

```
Q: Who composed the score for Interstellar?

Answer: "Hans Zimmer is a famous film composer known for many works."
         ← technically mentions Zimmer but doesn't answer the question
```

`answer_relevancy` will be low. Fix: tune the generation prompt.

### The key diagnostic question

When an answer is wrong, ask:
- Was the right document retrieved? → look at the chunks
- Was the right document retrieved but ignored? → look at faithfulness
- Was the answer just off-topic? → look at relevancy

Each root cause has a different fix.

---

## 1:35 — Exercise 2: Grade by hand

**Goal:** score 5 answers yourself — build intuition before trusting RAGAS numbers.

```bash
python exercises/02-grade-by-hand/exercise.py
```

For each of the 5 questions you will see:
- The retrieved chunks
- The RAG answer
- The ground truth

Score each on a 0–3 scale:
- **Faithfulness:** is the answer supported by the chunks?
- **Answer Relevancy:** does it answer the question?
- **Context Recall:** did retrieval surface the right document?

Fill in the SCORECARD at the bottom of the exercise file.

> **Key question:** when the answer was wrong — was the retrieved chunk wrong, or did the LLM go off-script? That distinction tells you where to fix.

Save your scores. In Exercise 3 you will compare them to RAGAS.

---

## 2:00 — Part 3: RAGAS

### What RAGAS does

RAGAS automates what you just did by hand.

It runs a small LLM internally (we point it at Claude haiku) to judge:
- Whether the answer is grounded in the context (faithfulness)
- Whether the answer is on-topic (answer_relevancy)
- Whether the retrieved context contained the ground-truth answer (context_recall)

### What RAGAS is not

- It is not magic. It has false positives.
- It is not a replacement for reading your outputs.
- It is a signal, not a verdict. A drop of 0.1 in faithfulness is worth investigating.

### Live demo — show RAGAS running

```bash
python exercises/03-ragas-scores/exercise.py
```

Walk through the output as it runs (~90 seconds).

Point out:
1. The per-question table — which questions score low?
2. The aggregate row — this is the number you track over time
3. The colour coding: green ≥ 0.7, yellow ≥ 0.4, red < 0.4

Ask the room: do the low-scoring questions match what they expected from Exercise 2?

---

## 2:15 — Exercise 3: RAGAS scores

**Goal:** run RAGAS on all 10 questions and compare to your hand scores.

```bash
python exercises/03-ragas-scores/exercise.py
```

This takes ~60–90 seconds. While it runs, look at the questions and make a prediction: which 2–3 will score lowest?

**When it finishes:**
- Do the RAGAS scores match your intuition from Exercise 2?
- Where do your hand scores and RAGAS scores disagree?
- Those disagreements are the most interesting cases.

---

## 2:30 — ☕ Short break

---

## 2:40 — Exercise 4: Break it, fix it

**Goal:** deliberately make the pipeline worse and watch the specific metrics drop.

```bash
python exercises/04-fix-and-rerun/exercise.py
```

Two breaks are pre-wired. You need to uncomment the TODO lines:

**Break A — Retrieval (top_k=1)**
- Change `top_k` from 3 to 1
- Only one chunk is retrieved per question
- Which metric drops? Why?

**Break B — Prompt (no grounding)**
- Remove the line "Answer using ONLY the context passages provided"
- The LLM is no longer told to stay within the retrieved docs
- Which metric drops this time? Is it the same one?

**Expected outcome:**

| Pipeline | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|-----------------|----------------|
| Baseline (top_k=3) | ~0.90 | ~0.88 | ~0.85 |
| Break A: top_k=1 | ~0.88 | ~0.85 | ~0.60 ↓ |
| Break B: no grounding | ~0.55 ↓ | ~0.85 | ~0.85 |

**The lesson:** different breaks hurt different metrics. That is the whole point of having three metrics — each one points at a different part of the pipeline.

**Stuck?** See `exercises/04-fix-and-rerun/solution.py`.

---

## 3:00 — Part 4: When scores are bad

### The diagnostic playbook

```
Score is low — what do I do?
│
├─ context_recall is low
│    → The right document was not retrieved
│    → Try: increase top_k
│    → Try: better embedding model (e.g. text-embedding-3-small)
│    → Try: add metadata filters (by date, by category)
│    → Try: hybrid search (vector + keyword)
│
├─ faithfulness is low
│    → The LLM is ignoring the context
│    → Try: strengthen the system prompt grounding instruction
│    → Try: use a more instruction-following model
│    → Try: reduce max_tokens so the LLM has less room to wander
│
└─ answer_relevancy is low
     → The answer is off-topic or vague
     → Try: be more specific in the generation prompt
     → Try: add few-shot examples of good answers
     → Try: ask the LLM to answer in one sentence only
```

### The eval loop you want

```
Change something
    ↓
Run RAGAS
    ↓
Did the score improve?
    ↓
Ship it or revert
```

This is how you iterate on a RAG system without guessing.

### What to track in production

- Run RAGAS on a fixed sample (e.g. 50 questions) after every change
- Alert if any metric drops more than 0.05 from baseline
- Log every retrieval + answer pair so you can audit failures manually

---

## 3:10 — End-to-end review + Q&A

### Live walkthrough

Run the full pipeline on a new question not in the eval set:

```bash
python pipeline/rag.py
```

Watch the retrieval happen in real time. Point out:
- Which chunks were retrieved
- The similarity scores
- What the answer looks like

Then ask: what would you change to make this better?

### What you built today

```
pipeline/corpus.py      — document store + ground-truth eval set
pipeline/rag.py         — embed → index → retrieve → generate

Exercise 1              — assembled the pipeline from scratch
Exercise 2              — graded answers by hand
Exercise 3              — automated grading with RAGAS
Exercise 4              — break-and-fix to understand each metric
```

### How RAGAS differs from LLM-as-judge with pure Python

A common alternative is to write your own judge: call the LLM, ask "is this answer faithful?", parse yes/no.

RAGAS does the same thing — but:
- The prompts are calibrated and versioned
- The scoring is normalised (0–1, not yes/no)
- context_recall uses a decomposition approach that is more robust than a single yes/no call
- The framework is maintained and improving

For most teams, RAGAS is the right starting point. Write your own judge when you have a domain-specific scoring criterion RAGAS cannot express.

### Next steps

- Add RAGAS to your CI pipeline: run on every PR that changes a prompt
- Expand the eval set: 10 questions is a start, 50–100 is production-grade
- Try a harder corpus: replace movie reviews with your own documents
- Explore other RAGAS metrics: `context_precision`, `answer_correctness`

---

*End of workshop. Repository: https://github.com/satyamsah/rag-evals-workshop*
