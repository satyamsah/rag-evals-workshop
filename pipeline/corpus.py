"""
pipeline/corpus.py — the movie review knowledge base.

This is the document store our RAG pipeline retrieves from.
25 short reviews covering 10 movies — small enough to inspect by eye,
big enough to have interesting retrieval failures.

In production this would be thousands of documents loaded from a database or S3.
The retrieval mechanism (vector search) is identical at any scale.
"""

DOCUMENTS = [
    # ── Inception ──────────────────────────────────────────────────────────
    {
        "id": "inc-1",
        "movie": "Inception",
        "text": "Inception is a 2010 sci-fi thriller directed by Christopher Nolan. "
                "The film follows Dom Cobb, a thief who steals secrets by entering people's dreams.",
    },
    {
        "id": "inc-2",
        "movie": "Inception",
        "text": "Leonardo DiCaprio plays Dom Cobb in Inception. "
                "The supporting cast includes Joseph Gordon-Levitt, Elliot Page, and Tom Hardy.",
    },
    {
        "id": "inc-3",
        "movie": "Inception",
        "text": "Inception won four Academy Awards including Best Cinematography and Best Visual Effects. "
                "It grossed over $836 million worldwide.",
    },
    # ── The Dark Knight ────────────────────────────────────────────────────
    {
        "id": "tdk-1",
        "movie": "The Dark Knight",
        "text": "The Dark Knight (2008) is a superhero film directed by Christopher Nolan. "
                "It is the second film in Nolan's Batman trilogy.",
    },
    {
        "id": "tdk-2",
        "movie": "The Dark Knight",
        "text": "Heath Ledger plays the Joker in The Dark Knight. "
                "Ledger won a posthumous Academy Award for Best Supporting Actor for the role.",
    },
    {
        "id": "tdk-3",
        "movie": "The Dark Knight",
        "text": "The Dark Knight grossed over $1 billion worldwide and is widely considered "
                "one of the greatest superhero films ever made.",
    },
    # ── Interstellar ───────────────────────────────────────────────────────
    {
        "id": "int-1",
        "movie": "Interstellar",
        "text": "Interstellar (2014) is a sci-fi epic directed by Christopher Nolan. "
                "The story follows a team of astronauts travelling through a wormhole near Saturn.",
    },
    {
        "id": "int-2",
        "movie": "Interstellar",
        "text": "Matthew McConaughey stars as Cooper in Interstellar. "
                "Anne Hathaway, Jessica Chastain, and Michael Caine also appear in the film.",
    },
    {
        "id": "int-3",
        "movie": "Interstellar",
        "text": "Hans Zimmer composed the score for Interstellar. "
                "The film won an Academy Award for Best Visual Effects.",
    },
    # ── The Matrix ─────────────────────────────────────────────────────────
    {
        "id": "mat-1",
        "movie": "The Matrix",
        "text": "The Matrix (1999) is a sci-fi action film directed by the Wachowski sisters. "
                "It stars Keanu Reeves as Neo, a hacker who discovers reality is a simulation.",
    },
    {
        "id": "mat-2",
        "movie": "The Matrix",
        "text": "The Matrix won four Academy Awards including Best Visual Effects and Best Film Editing. "
                "It revolutionised action filmmaking with its bullet-time photography technique.",
    },
    # ── Parasite ───────────────────────────────────────────────────────────
    {
        "id": "par-1",
        "movie": "Parasite",
        "text": "Parasite (2019) is a South Korean black comedy thriller directed by Bong Joon-ho. "
                "It explores class conflict between two families.",
    },
    {
        "id": "par-2",
        "movie": "Parasite",
        "text": "Parasite became the first non-English language film to win the Academy Award for Best Picture. "
                "It also won Best Director, Best Original Screenplay, and Best International Feature Film.",
    },
    # ── Everything Everywhere All at Once ─────────────────────────────────
    {
        "id": "ee-1",
        "movie": "Everything Everywhere All at Once",
        "text": "Everything Everywhere All at Once (2022) is directed by the Daniels (Daniel Kwan and Daniel Scheinert). "
                "Michelle Yeoh plays Evelyn, a laundromat owner caught in a multiverse adventure.",
    },
    {
        "id": "ee-2",
        "movie": "Everything Everywhere All at Once",
        "text": "Everything Everywhere All at Once won seven Academy Awards including Best Picture, "
                "Best Director, and Best Actress for Michelle Yeoh.",
    },
    # ── Oppenheimer ────────────────────────────────────────────────────────
    {
        "id": "opp-1",
        "movie": "Oppenheimer",
        "text": "Oppenheimer (2023) is a biographical thriller directed by Christopher Nolan. "
                "It chronicles J. Robert Oppenheimer's role in developing the first nuclear weapon.",
    },
    {
        "id": "opp-2",
        "movie": "Oppenheimer",
        "text": "Cillian Murphy plays J. Robert Oppenheimer. The cast also includes Emily Blunt, "
                "Matt Damon, and Robert Downey Jr.",
    },
    {
        "id": "opp-3",
        "movie": "Oppenheimer",
        "text": "Oppenheimer won seven Academy Awards in 2024 including Best Picture, Best Director, "
                "Best Actor for Cillian Murphy, and Best Supporting Actor for Robert Downey Jr.",
    },
    # ── Get Out ────────────────────────────────────────────────────────────
    {
        "id": "go-1",
        "movie": "Get Out",
        "text": "Get Out (2017) is a psychological horror film written and directed by Jordan Peele. "
                "It follows a Black man who visits his white girlfriend's family estate.",
    },
    {
        "id": "go-2",
        "movie": "Get Out",
        "text": "Get Out won the Academy Award for Best Original Screenplay for Jordan Peele. "
                "Daniel Kaluuya stars as Chris Washington in the film.",
    },
    # ── Dune ───────────────────────────────────────────────────────────────
    {
        "id": "dun-1",
        "movie": "Dune: Part One",
        "text": "Dune: Part One (2021) is a sci-fi epic directed by Denis Villeneuve. "
                "It adapts the first half of Frank Herbert's 1965 novel.",
    },
    {
        "id": "dun-2",
        "movie": "Dune: Part One",
        "text": "Timothée Chalamet plays Paul Atreides in Dune. "
                "The film won six Academy Awards including Best Cinematography.",
    },
    # ── Mad Max: Fury Road ─────────────────────────────────────────────────
    {
        "id": "mm-1",
        "movie": "Mad Max: Fury Road",
        "text": "Mad Max: Fury Road (2015) is an action film directed by George Miller. "
                "It stars Tom Hardy as Max Rockatansky and Charlize Theron as Imperator Furiosa.",
    },
    {
        "id": "mm-2",
        "movie": "Mad Max: Fury Road",
        "text": "Mad Max: Fury Road won six Academy Awards including Best Film Editing. "
                "It is set in a post-apocalyptic desert wasteland.",
    },
]


# ── Evaluation dataset ────────────────────────────────────────────────────
#
# 10 question-answer pairs used to score the pipeline.
# Each entry has:
#   question        — what the user asks
#   ground_truth    — the correct answer (used by RAGAS for context_recall)
#   reference_docs  — which doc IDs contain the answer (for inspection)
#
EVAL_QUESTIONS = [
    {
        "question": "Who directed Inception?",
        "ground_truth": "Inception was directed by Christopher Nolan.",
        "reference_docs": ["inc-1"],
    },
    {
        "question": "Which actor won an Academy Award for playing the Joker?",
        "ground_truth": "Heath Ledger won a posthumous Academy Award for Best Supporting Actor for playing the Joker in The Dark Knight.",
        "reference_docs": ["tdk-2"],
    },
    {
        "question": "What was the first non-English film to win Best Picture?",
        "ground_truth": "Parasite was the first non-English language film to win the Academy Award for Best Picture.",
        "reference_docs": ["par-2"],
    },
    {
        "question": "Who plays Evelyn in Everything Everywhere All at Once?",
        "ground_truth": "Michelle Yeoh plays Evelyn in Everything Everywhere All at Once.",
        "reference_docs": ["ee-1"],
    },
    {
        "question": "How many Oscars did Oppenheimer win?",
        "ground_truth": "Oppenheimer won seven Academy Awards in 2024.",
        "reference_docs": ["opp-3"],
    },
    {
        "question": "Who wrote and directed Get Out?",
        "ground_truth": "Get Out was written and directed by Jordan Peele.",
        "reference_docs": ["go-1"],
    },
    {
        "question": "What novel is Dune: Part One based on?",
        "ground_truth": "Dune: Part One is based on Frank Herbert's 1965 novel Dune.",
        "reference_docs": ["dun-1"],
    },
    {
        "question": "Who composed the score for Interstellar?",
        "ground_truth": "Hans Zimmer composed the score for Interstellar.",
        "reference_docs": ["int-3"],
    },
    {
        "question": "What filming technique did The Matrix pioneer?",
        "ground_truth": "The Matrix pioneered bullet-time photography.",
        "reference_docs": ["mat-2"],
    },
    {
        "question": "Who plays Furiosa in Mad Max: Fury Road?",
        "ground_truth": "Charlize Theron plays Imperator Furiosa in Mad Max: Fury Road.",
        "reference_docs": ["mm-1"],
    },
]
