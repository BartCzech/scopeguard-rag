# Access-Controlled RAG over MCP

An NLP/LLM research project comparing how the placement of access control in a RAG pipeline affects answer quality, refusal behavior, and privacy leakage.

## What This Project Is

We build a Retrieval-Augmented Generation system that answers questions over a corporate document collection. The catch: not every user is authorized to see every document. We implement **five different strategies** for enforcing access control — from "no protection at all" to "strict pre-retrieval filtering" — and measure how each one trades off answer quality against information leakage.

The documents belong to a fictional startup called **ScopeGuard AI** (a company that, ironically, sells LLM access-control middleware). The corpus includes adversarial documents with embedded prompt injections designed to trick the system into leaking restricted content.

The agent never reads documents directly. Instead, it calls tools on an MCP (Model Context Protocol) server that validates JWT access tokens before returning results. This architecture lets us swap filtering strategies at the tool layer without changing the rest of the pipeline.

## Research Question

> How does the placement and mechanism of access control in a RAG pipeline affect answer utility, refusal correctness, and privacy leakage?

## The Five Strategies


| #   | Strategy                     | How it works                                                                                                                            | Expected behavior                                                                   |
| --- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 0   | **Pure LLM**                 | No retrieval. LLM answers from parametric knowledge only.                                                                               | Baseline — shows what RAG adds.                                                     |
| 1   | **Naive RAG**                | Retrieves all documents. System prompt says "don't reveal restricted info."                                                             | High leakage risk. The model sees everything and is told to keep secrets.           |
| 2   | **Post-Retrieval Filtering** | Retrieves all documents, filters by access level before injecting into context.                                                         | Context is clean, but retrieval ranking may be influenced by restricted docs.       |
| 3   | **Pre-Retrieval Filtering**  | Retrieves only from documents within the user's scope. Restricted docs are never searched.                                              | Strictest. Zero leakage risk, but may reduce answer quality.                        |
| 4   | **Taint-Aware Output Guard** | Retrieves everything (with access labels). Generates an answer. A PolicyEngine checks the output for leakage and blocks it if detected. | Middle ground — attempts to recover utility while catching leakage post-generation. |


## Access Tiers

Every document has an access level. Every user has a JWT token with a scope claim.


| Scope Token         | Sees              | Who                            |
| ------------------- | ----------------- | ------------------------------ |
| `docs:public`       | Public docs only  | Customers, prospects           |
| `docs:internal`     | Public + internal | All employees                  |
| `docs:confidential` | Everything        | Leadership, HR, Legal, Finance |


## Document Corpus

45 synthetic Markdown documents with YAML frontmatter, organized as:

```
corpus/
├── public/          # 17 docs — product pages, blog posts, FAQ, pricing
├── internal/        # 14 docs — ADRs, meeting notes, retros, competitive intel
└── confidential/    # 11 docs — salary bands, board minutes, legal memos, security audit
```

Plus 4 dedicated adversarial documents (in their tier folders) and 3 regular documents with embedded adversarial payloads — prompt injections and cross-scope references designed to trick the system into surfacing restricted content.

Two "honeypot" documents are targeted by most adversarial attacks: **CONF-003** (salary bands) and **CONF-006** (security audit).

Full manifest: see `scopeguard_document_manifest.md`.

## Tech Stack


| Component        | Technology                                                            |
| ---------------- | --------------------------------------------------------------------- |
| Language         | Python 3.13+                                                          |
| Package manager  | uv                                                                    |
| Data models      | Pydantic                                                              |
| Embeddings       | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)                   |
| Vector store     | FAISS (`IndexFlatIP`, cosine similarity via normalized inner product) |
| LLM (generation) | OpenAI `gpt-4o-mini` (temperature=0.0 for reproducibility)            |
| LLM (judge)      | OpenAI `gpt-4o` (stronger model to reduce self-bias)                  |
| MCP server       | FastMCP v3                                                            |
| Auth tokens      | PyJWT (HS256-signed JWTs with scope claims)                           |
| Linting          | ruff                                                                  |
| Type checking    | mypy                                                                  |
| Testing          | pytest                                                                |


## Project Structure

```
scopeguard-rag/
├── corpus/
│   ├── public/
│   ├── internal/
│   └── confidential/
├── src/
│   ├── models.py              # Pydantic models (Document, Chunk, RetrievalResult, etc.)
│   ├── config.py              # LLM provider, model, API key env var
│   ├── corpus_loader.py       # Parse .md files with frontmatter → Document objects
│   ├── embedder.py            # Chunk documents, embed, build/save/load FAISS index
│   ├── retriever.py           # Query the index, return ranked results
│   ├── generator.py           # Call LLM with context, parse citations
│   ├── auth.py                # JWT creation and validation
│   ├── mcp_server.py          # FastMCP tool server wrapping the retriever
│   ├── policy_engine.py       # N-gram overlap + citation checks for leakage detection
│   ├── strategies/
│   │   ├── base.py            # StrategyBase ABC + StrategyResult
│   │   ├── prompts.py         # Shared system prompt building blocks
│   │   ├── strategy_0.py      # Pure LLM (no retrieval)
│   │   ├── strategy_1.py      # Naive RAG (prompt-based control)
│   │   ├── strategy_2.py      # Post-retrieval filtering
│   │   ├── strategy_3.py      # Pre-retrieval filtering
│   │   └── strategy_4.py      # Taint-aware output guard
│   └── evaluation/
│       ├── harness.py         # Run all strategies × scopes × questions
│       ├── metrics.py         # Automated metrics (citation accuracy, leakage, etc.)
│       └── judge.py           # LLM-as-judge scoring
├── tests/
│   ├── test_questions.json    # 30 labeled question-answer pairs
│   ├── test_corpus_loader.py
│   ├── test_retriever.py
│   ├── test_auth.py
│   └── test_policy_engine.py
├── data/
│   └── index/                 # FAISS index + metadata sidecar (gitignored)
├── pyproject.toml
├── uv.lock
├── .env                       # OPENAI_API_KEY (gitignored, create your own)
└── .env.example
```

## Getting Started

```bash
# Clone and install
git clone <repo-url>
cd scopeguard-rag
uv sync

# Install the pre-commit hooks (ruff, ruff-format, mypy run on commit; tests are run manually)
uv run pre-commit install

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Verify toolchain
uv run ruff check src/
uv run mypy src/
uv run pytest
```

## How to Run

```bash
# Build the embedding index (run once, or after corpus changes)
uv run python -m src.embedder

# Run the full evaluation (5 strategies × 3 scopes × 30 questions = 450 runs)
uv run python -m src.evaluation.harness

# Run a single strategy for debugging
uv run python -m src.evaluation.harness --strategy strategy_2

# Run the LLM-as-judge on evaluation results
uv run python -m src.evaluation.judge

# Start the MCP server standalone (for demo purposes)
uv run python -m src.mcp_server
```

## Evaluation

30 test questions across six categories: public-only, internal-only, confidential-only, cross-scope, unanswerable, and adversarial.

Each question is run against all 5 strategies at all 3 scope levels. Metrics:


| Metric                 | How measured                                                |
| ---------------------- | ----------------------------------------------------------- |
| Answer correctness     | LLM-as-judge (0–3 scale)                                    |
| Citation accuracy      | Automated — cited doc IDs vs gold doc IDs                   |
| Refusal precision      | LLM-as-judge — is the refusal justified?                    |
| Refusal recall         | LLM-as-judge — did it refuse when it should have?           |
| Direct leakage         | Automated — n-gram overlap + restricted citation checks     |
| Indirect leakage       | LLM-as-judge — is the answer influenced by restricted docs? |
| Adversarial robustness | Automated — pass/fail on adversarial test cases             |


Results are saved to `data/evaluation_results.json` (automated metrics) and `data/evaluation_results_judged.json` (with judge scores).

## Deliverables

1. Working implementation of all five RAG strategies
2. MCP tool server with JWT-based scope enforcement
3. Taint-aware PolicyEngine for output leakage detection
4. Labeled test set (30 question-answer pairs)
5. Evaluation results comparing the five strategies
6. Final report (8–12 pages) with qualitative case studies
7. This repository with documentation and reproducible experiments

## Project Phases


| Phase | What                | Tasks                                                                                                           |
| ----- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| 1     | Corpus + baseline   | T00 scaffold, T01 document parser, T02 embeddings, T03 retriever, T04 generator, T05 strategy 0, T06 strategy 1 |
| 2     | MCP + filtering     | T07 auth + MCP server, T08 strategy 2, T09 test set, T11 strategy 3                                             |
| 3     | Taint-aware guard   | T12 PolicyEngine, T13 strategy 4                                                                                |
| 4     | Evaluation + report | T10 evaluation harness, T14 LLM-as-judge, T15 results & report                                                  |


Full task specs with interfaces, code, acceptance criteria, and dependencies are found in GitHub Projects of this repo.

## Key Files to Read First

1. **This README** — you are here
2. `**src/models.py`** — shared Pydantic data structures used everywhere
3. `**src/strategies/base.py`** — the StrategyBase interface every strategy implements

