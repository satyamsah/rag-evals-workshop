# RAG Evals with RAGAS: Build, Score, and Fix Your Pipeline

### Workshop guide — follow this top to bottom during the session

---

## Schedule at a glance

| Time | Block |
|------|-------|
| 0:00 | Setup & welcome |
| 0:15 | Part 1 — RAG, the RAG pipeline, and why we evaluate it |
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

```text
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

## 0:15 — Part 1: RAG, the RAG Pipeline, and Why We Evaluate It

### First: what is RAG?

Before we talk about evaluation, let's make sure we understand the system we
are evaluating.

A **large language model** is trained on a large amount of data, but it does not
automatically know the **latest information** or the **private information** inside
your organization.

For example, imagine we ask:

> **"According to our company's internal policy, how many days of parental leave are available?"**

The answer may not exist in the model's training data at all.

RAG — **Retrieval-Augmented Generation** — gives the LLM access to relevant
information at question time.

The basic idea is simple:

```text
Question
   ↓
Find relevant information
   ↓
Give that information to the LLM
   ↓
Generate an answer
```

Instead of asking the LLM to answer entirely from what it learned during
training, we first **retrieve relevant documents** and provide them as **context**.

---

### What happens inside a RAG system?

Before somebody asks a question, we first need to **prepare our documents** so
that they can be **searched efficiently**.

Think about a RAG system as having two major stages:

1. **Preparing the knowledge base**
2. **Answering the user's question**

---

### Part 1 — Preparing the knowledge base

The first stage is preparing the documents.

The basic flow is:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Index
```

Let's look at each step.

### Step 1 — Documents

We start with the information we want our RAG system to answer questions
from.

These could be:

- Company documents
- Product documentation
- PDFs
- Knowledge-base articles
- Internal policies
- Database records
- Any other information we want the LLM to use

The important thing is that this information is available to our system even
if it was **not part of the LLM's original training data**.

### Step 2 — Chunking

Large documents are usually split into smaller pieces called **chunks**.

Why do we do this?

Because when a user asks a question, we usually don't want to send an **entire
large document** to the LLM.

We want to find the **smaller section** that is actually **relevant to the question**.

```text
Large document
       ↓
 ┌─────────────┐
 │ Chunk 1     │
 ├─────────────┤
 │ Chunk 2     │
 ├─────────────┤
 │ Chunk 3     │
 ├─────────────┤
 │ Chunk 4     │
 └─────────────┘
```

Later, when somebody asks a question, we can retrieve only the chunks that
are most relevant.

### Step 3 — Embeddings

Now we need a way to search these chunks based on their meaning.

This is where **embeddings** come in.

An embedding converts text into a **numerical representation** — a **vector**.

For example:

```text
"Christopher Nolan directed Inception"
                ↓
       [0.12, -0.43, 0.87, ...]
```

We don't need to understand every number in the vector.

The important intuition is:

> **Texts with similar meanings tend to be represented by vectors that are close to each other.**

For example, imagine our document contains:

> "Inception is a 2010 science-fiction thriller directed by Christopher Nolan."

And the user asks:

> **"Who directed Inception?"**

The **wording** in the question is not exactly the same as the wording in the
document.

Embeddings allow us to compare the **meaning** of the question with the meaning
of the document chunks and find relevant chunks even when the **wording is
different**.

That is the important intuition behind **semantic search**.

### Step 4 — Indexing

Once we have converted our document chunks into vectors, we need to **store
them** somewhere that allows us to **search efficiently**.

That is where a vector index comes in.

In this workshop, we use **FAISS — Facebook AI Similarity Search**.

Conceptually:

```text
Document chunks
      ↓
Embeddings
      ↓
Vector index
      ↓
Efficient similarity search
```

When a user asks a question, we also create an **embedding for the question**
and search the index for the **closest vectors**.

Those closest chunks become our **retrieved context**.

---

### Part 2 — Answering the question

Now that our knowledge base is prepared, a user can ask a question.

The answering process looks like this:

```text
Question
   ↓
Create question embedding
   ↓
Retrieve relevant chunks
   ↓
Retrieved context
   ↓
LLM
   ↓
Answer
```

So the core RAG flow is:

**Retrieve → Context → Generate**

Let's make this concrete.

Suppose the user asks:

> **"Who directed Inception?"**

The retriever searches our index and finds:

```text
Chunk 1:
"Inception is a 2010 science-fiction thriller
directed by Christopher Nolan."

Chunk 2:
"Inception received several Academy Award
nominations."

Chunk 3:
"Christopher Nolan has directed several
successful films."
```

The retrieved chunks are provided to the LLM as context.

The LLM can then generate:

> "Inception was directed by Christopher Nolan."

That is the basic RAG system we are going to evaluate.

---

## So where does evaluation fit?

This is an important distinction:

> **Evaluation is not another step inside the RAG pipeline. It is a way of measuring how well the pipeline is working.**

Our RAG system produces an answer.

Evaluation helps us understand:

- Did we retrieve the right information?
- Did the LLM use the retrieved information correctly?
- Did the answer actually answer the user's question?

Think of it as a diagnostic layer around our RAG system:

```text
                    RAG SYSTEM

Question → Retrieval → Context → LLM → Answer
              │                    │
              └──── Evaluation ───┘
```

The RAG system creates the answer.

The evaluation system tells us how well the RAG system performed.

---

## What can go wrong?

Before looking at individual failures, there is one important distinction:

### Symptoms vs. Root Causes

When we evaluate a RAG system, a metric tells us what we are seeing from the
outside — it does not necessarily tell us the exact root cause.

For example, if **Context Recall is low** (we will define this properly in a moment — for now think of it as the retrieval signal), the problem could be:

- The documents were not ingested correctly
- The chunks were poorly created
- The embeddings did not capture the meaning well
- `top_k` was too small
- The retrieval configuration was not appropriate

So think of the metric as a **signal**.

The actual root cause may be somewhere earlier in the RAG pipeline:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Index
   ↓
Retrieval
   ↓
Context
   ↓
LLM
   ↓
Answer
   ↓
Evaluation
```

By the end of the workshop, you will know exactly which part of the pipeline to look at when a score drops.

The important idea for now is:

> **The metric tells us where to investigate. We then inspect the pipeline to find the root cause.**

That distinction will become important when we look at the three failure cases.

### Failure 1 — We retrieved the wrong information

User asks:

> **"Who plays Evelyn in Everything Everywhere All at Once?"**

But our retriever returns:

> "Everything Everywhere All at Once won seven Academy Awards."

This is related to the movie, but it does **not contain the information needed** to
answer the question.

This is a **retrieval symptom**. The root cause could be **retrieval itself**, or something earlier such as **chunking**, **embeddings**, indexing, or ingestion.

### Failure 2 — We retrieved the right information, but the LLM ignored it

User asks:

> **"How many Oscars did Oppenheimer win?"**

Retrieved context says:

> "Oppenheimer won seven Academy Awards in 2024."

But the LLM answers:

> "Oppenheimer won 13 Academy Awards."

This time, retrieval worked.

The correct information was already available to the LLM.

The problem is that the **generated answer was not grounded** in the retrieved
context.

### Failure 3 — The answer is related, but doesn't actually answer the question

User asks:

> **"Who composed the score for Interstellar?"**

The LLM answers:

> "Hans Zimmer is a famous film composer known for many works."

This mentions the right person, but it **doesn't directly answer the question**.

The answer should have been:

> "Hans Zimmer composed the score for Interstellar."

So here the problem is with the **relevance of the answer**.

---

## The three questions behind RAG evaluation

These three failure cases lead us to three important evaluation questions:

1. **Did we retrieve the information needed to answer the question?**
2. **Did the answer stay supported by the retrieved context?**
3. **Did the answer actually address the user's question?**

These map to the three metrics we will use.

| What are we asking? | Metric | What does it tell us? |
|---|---|---|
| Did we retrieve the information needed to answer the question? | **Context Recall** | Whether the relevant information was present in the retrieved context |
| Did the answer stay supported by the retrieved context? | **Faithfulness** | Whether the LLM stayed grounded in the retrieved information |
| Did the answer actually address the user's question? | **Answer Relevancy** | Whether the generated answer is relevant to what the user asked |

### Context Recall

**Context Recall** asks:

> **Did our retrieval process find the information needed to answer the question?**

If the answer is in our document collection but our retriever **fails to
surface the relevant document or chunk**, Context Recall will be low.

Things that can cause this include:

- `top_k` being too small
- Poor embeddings
- Vocabulary mismatch
- Poor chunking
- Missing metadata filters

### Faithfulness

**Faithfulness** asks:

> **Is the generated answer supported by the retrieved context?**

Imagine the retrieved context says:

> "Oppenheimer won seven Academy Awards."

But the LLM says:

> "Oppenheimer won 13 Academy Awards."

The **correct document was retrieved**, but the answer is **not supported** by that
document.

That is a faithfulness problem.

### Answer Relevancy

**Answer Relevancy** asks:

> **Does the answer actually address the user's question?**

For example:

Question:

> "Who composed the score for Interstellar?"

Answer:

> "Hans Zimmer is a famous composer."

The answer is related, but it doesn't directly answer the question.

A more relevant answer is:

> "Hans Zimmer composed the score for Interstellar."

---

## Why not just read the answers?

You just learned to spot three problems in any RAG answer by eye — wrong retrieval, ignored context, off-topic generation. That works well for a handful of questions.

If we have five questions, we can read the answers ourselves.

But imagine we have:

- 500 questions
- A new embedding model to compare
- A changed prompt
- A new retrieval strategy

We need a **repeatable way** to tell whether things got better or worse.

That is where **RAGAS** comes in — and we will get to it in Part 3.

First, let's build the pipeline.

---

## 0:40 — Exercise 1: Build the pipeline

**Goal:** assemble the three pieces of a RAG pipeline from scratch.

```text
What you are building:

  Documents ──► build_index() ──► FAISS index
                                       │
  Question  ──► retrieve()    ──► top-3 chunks
                                       │
  Question + chunks ──► generate() ──► Answer
```

```bash
python exercises/01-build-rag/exercise.py
```

**Three functions to fill in:**

| Function | What it does |
|---|---|
| `build_index(documents)` | Embed all 25 movie passages, store in FAISS |
| `retrieve(question, index, texts)` | Embed the question, find top-3 nearest chunks |
| `generate(question, chunks)` | Pass question + chunks to Claude, return answer |

**What you should see when it works:**

```text
✓ Index built: 25 vectors

Q: Who directed Inception?
  chunk 1 (score 0.72): Inception is a 2010 sci-fi thriller directed by Christopher Nolan...
Answer: Inception was directed by Christopher Nolan.
```

**Stuck?** See `exercises/01-build-rag/solution.py`.

> **Presenter note:** Before releasing the room, run `encoder.encode(["hello world"])` live.
> Show the array of numbers. Ask: "what makes two vectors close together?"
> Answer: similar meaning. That is the whole foundation of retrieval.

---

## 1:00 — ☕ Break

---

## 1:15 — Part 2: Grading a RAG answer — what does "good" mean?

In Part 1 we learned the three ways a RAG answer can fail.

Now we are going to make that concrete by looking at real answers and asking:
*which failure is this?*

Before we use a tool to score answers automatically, we need to build the
intuition ourselves — because if you cannot spot a bad answer by eye, you
cannot trust a score you do not understand.

### The three questions to ask for any answer

```text
Step 1 — Look at the retrieved chunks
         Did they contain the information needed to answer the question?
                        │
              YES       │        NO
               │        │         │
               ▼        │         ▼
Step 2 —  Look at      │    → Context Recall problem
          the answer   │      The right document was never retrieved
               │        │
               ▼        │
         Does the answer match the retrieved chunks?
                        │
              YES       │        NO
               │        │         │
               ▼        │         ▼
Step 3 —  Does it      │    → Faithfulness problem
          directly     │      The LLM ignored the context
          answer the   │
          question?    │
               │        │
              NO        │
               │        │
               ▼        │
         → Answer Relevancy problem
           The answer is vague or off-topic
```

Use this as your mental checklist in Exercise 2.

### Quick reference

| What you observe | Signal | Where to look |
|---|---|---|
| Right doc not retrieved | Context Recall ↓ | Chunking, embeddings, top_k |
| Answer contradicts the retrieved chunk | Faithfulness ↓ | System prompt grounding |
| Answer is vague or off-topic | Answer Relevancy ↓ | Generation prompt |

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

RAGAS automates what you just did by hand in Exercise 2.

It runs a small LLM internally (we point it at Claude haiku) to judge:

- Whether the answer is grounded in the context → **Faithfulness**
- Whether the answer is on-topic → **Answer Relevancy**
- Whether the retrieved context contained the ground-truth answer → **Context Recall**

Each score is a number from **0 to 1**. **Higher is better.** A score **below 0.7** is worth investigating.

Think of RAGAS as a **repeatable version of your Exercise 2 scorecard** — same questions, same criteria, but it can run on **500 answers in 90 seconds** instead of you reading them one by one.

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

**Goal:** run RAGAS on all 10 questions and compare to your hand scores from Exercise 2.

```text
What RAGAS is doing internally:

  Question + Answer + Retrieved chunks + Ground truth
                        │
                        ▼
              ┌─────────────────────┐
              │   Claude haiku      │  ← RAGAS uses an LLM as the judge
              │   (as judge)        │
              └─────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Faithfulness  Answer         Context
    0.0 – 1.0     Relevancy      Recall
                  0.0 – 1.0      0.0 – 1.0
```

```bash
python exercises/03-ragas-scores/exercise.py
```

This takes ~60–90 seconds. While it runs, predict: which 2–3 questions will score lowest?

**When it finishes:**

- Do the RAGAS scores match your intuition from Exercise 2?
- Where do your hand scores and RAGAS disagree?
- Those disagreements are the most interesting cases — they show the limits of automated scoring.

---

## 2:30 — ☕ Short break

---

## 2:40 — Exercise 4: Break it, fix it

**Goal:** deliberately make the pipeline worse and watch the specific metrics drop.

```bash
python exercises/04-fix-and-rerun/exercise.py
```

Two breaks are pre-wired. You need to uncomment the TODO lines.

### Break A — Retrieval (`top_k=1`)

- Change `top_k` from 3 to 1
- Only one chunk is retrieved per question
- Which metric drops? Why?

### Break B — Prompt (no grounding)

- Remove the line "Answer using ONLY the context passages provided"
- The LLM is no longer told to stay within the retrieved docs
- Which metric drops this time? Is it the same one?

### Expected outcome

| Pipeline | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|-----------------|----------------|
| Baseline (top_k=3) | ~0.90 | ~0.88 | ~0.85 |
| Break A: top_k=1 | ~0.88 | ~0.85 | ~0.60 ↓ |
| Break B: no grounding | ~0.55 ↓ | ~0.85 | ~0.85 |

**The lesson:** different breaks **hurt different metrics**. That is the whole point of having three metrics — each one **points at a different part of the pipeline**.

**Stuck?** See `exercises/04-fix-and-rerun/solution.py`.

---

## 3:00 — Part 4: When scores are bad

### The diagnostic playbook

```text
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

```text
Change something
    ↓
Run RAGAS
    ↓
Did the score improve?
    ↓
Ship it or revert
```

This is how you **iterate on a RAG system without guessing**.

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

Watch the retrieval happen in real time.

Point out:

- Which chunks were retrieved
- The similarity scores
- What the answer looks like

Then ask: what would you change to make this better?

---

### What you built today

```text
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

For most teams, **RAGAS is the right starting point**. Write your own judge when you have a **domain-specific scoring criterion** RAGAS cannot express.

### Next steps

- Add RAGAS to your CI pipeline: run on every PR that changes a prompt
- Expand the eval set: 10 questions is a start, 50–100 is production-grade
- Try a harder corpus: replace movie reviews with your own documents
- Explore other RAGAS metrics: `context_precision`, `answer_correctness`

---

*End of workshop. Repository: https://github.com/satyamsah/rag-evals-workshop*
