# AeroAgent

**A local-first, agentic AI framework built with asynchronous Python.**

AeroAgent is a from-scratch agent orchestration engine — no LangChain, no CrewAI. It routes tasks, calls tools, remembers past interactions, and traces every decision it makes, and it runs entirely on your machine for free using [Ollama](https://ollama.com/).

This is not a notebook experiment. It's built like a real system: type-safe state, defensive parsing around flaky LLM output, persistent memory, and a path to a containerized, deployable service.

| Phase | Status |
|--------|--------|
| Phase 1 — Core Loop & Tooling | Complete |
| Phase 2 — Memory & Context | Complete |
| Phase 3 — Evaluation & Observability | Complete |
| Phase 4 — Productionization | Complete |
| Phase 5 — Domain Specialization | Future |

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Roadmap](#roadmap)
- [Known Limitations and Mitigations](#known-limitations-and-mitigations)
- [License](#license)

---

## Why This Project Exists

Most "AI agent" tutorials wrap an API call in a `while` loop and call it a day. AeroAgent is an attempt to build the real thing: a system that can reason about a task, decide which tool to use, execute it safely, remember the outcome, and explain why it made each decision—all without depending on a paid API or a heavyweight framework.

The goal is to demonstrate **AI Systems Engineering**, not prompt engineering.

Core engineering principles:

- **Build** — Clean, type-safe, asynchronous Python (`asyncio`, `Pydantic v2`)
- **Debug** — Structured execution traces and an automated test suite
- **Explain** — Strict JSON schemas with graceful fallbacks when small local models fail to follow them
- **Deploy**  — Containerized deployment with a documented REST API, verified on every push via CI

---

## How It Works

AeroAgent enforces a strict, deterministic workflow around the LLM rather than trusting a small local model to plan freely, because 1–3B parameter models are unreliable planners. The execution loop works as follows:

1. **Memory first.** On every prompt, the agent is instructed (and code-enforced) to call `search_memory` before anything else to determine whether it already knows the answer.

2. **Web search only if needed.** If memory returns no relevant information, the agent is allowed exactly one `web_search` call. If memory does return useful information, `web_search` is blocked in code—even if the LLM attempts to call it.

3. **Automatic persistence.** Whenever `web_search` runs, its results are automatically saved using `save_to_memory`. This happens deterministically in Python rather than as an LLM decision, ensuring the model cannot forget to persist newly acquired information.

4. **Finalize.** Once results are available (from memory or web search), the agent must output a `finalize` action containing a `final_answer`. If it attempts to finalize before calling `search_memory`, the agent blocks the action and forces the memory lookup first.

5. **Defensive parsing.** Small local models frequently produce malformed output, so the agent includes several safeguards:
   - Retries once if the LLM's JSON cannot be parsed.
   - Infers a missing `"action"` key from available fields (for example, a `tool_name` implies `call_tool`).
   - Blocks duplicate tool calls using the same query.
   - Falls back to a synthesis step—a second plain-text LLM call over everything gathered so far—if the model finalizes without an answer, becomes stuck reasoning, or exhausts the maximum number of execution steps without producing a final response.

6. **Execution tracing.** Every step (`think`, `call_tool`, `error`, and `finalize`) is recorded in the `AgentState` and written to a JSON trace for debugging and inspection.

### Workflow

```text
You
 │
 ▼
CLI (main.py)
 │
 ▼
Agent Loop
 │
 ├── search_memory (forced first)
 │         │
 │         ├── Hit ───────────────► Finalize
 │         │
 │         └── Miss
 │                │
 │                ▼
 │         web_search (max once)
 │                │
 │                ▼
 │        save_to_memory (automatic)
 │                │
 │                ▼
 └──────────────► Finalize

JSON Trace + Final Answer
```

---

## Project Structure

The repository structure (`git ls-files`) is shown below. Runtime-generated directories are listed separately.

```text
 .
 ├── LICENSE
 ├── README.md
 ├── Dockerfile
 ├── docker-compose.yml
 ├── .dockerignore
 ├── .github/
 │   └── workflows/
 │       └── ci.yml              # Lint, test, and Docker build on push/PR
 ├── main.py                    # CLI entry point
 ├── pyproject.toml             # Package metadata, deps, ruff/pytest config
 ├── requirements.txt           # Pinned deps (see note below)
 ├── src/aeroagent/
 │   ├── __init__.py
 │   ├── agent.py                # Async state machine and defensive routing
 │   ├── llm.py                  # Ollama async client
 │   ├── state.py                # Pydantic v2 models (AgentState, ToolSchema)
 │   ├── tracer.py                # Local JSON execution logger
 │   ├── api/
 │   │   ├── __init__.py
 │   │   ├── main.py              # FastAPI app: /run, /health, rate limiting
 │   │   └── schemas.py           # Request/response models
 │   ├── observability/
 │   │   ├── __init__.py
 │   │   └── metrics.py           # Latency, token, and cost tracking
 │   ├── prompts/
 │   │   ├── __init__.py
 │   │   └── registry.py          # Versioned system prompts
 │   ├── evals/
 │   │   ├── __init__.py
 │   │   ├── dataset.py           # EvalCase model and starter dataset
 │   │   ├── judge.py             # LLM-as-a-judge scoring
 │   │   └── runner.py            # Eval suite runner and report generator
 │   ├── memory/
 │   │   ├── __init__.py
 │   │   ├── embedder.py          # Sentence-transformers wrapper
 │   │   ├── memory.py            # MemoryManager composition
 │   │   └── store.py             # ChromaDB read/write
 │   └── tools/
 │       ├── __init__.py
 │       ├── memory.py            # `save_to_memory` and `search_memory` tools
 │       ├── registry.py          # Type-safe tool execution wrapper
 │       └── search.py            # Async DuckDuckGo search tool
 └── tests/
     ├── __init__.py
     ├── test_agent.py            # Agent init, tool routing, defensive parsing
     ├── test_agent_loop.py       # Agent loop behavior
     ├── test_evals.py            # Eval harness offline tests
     └── test_memory.py           # Memory stack tests
```
Generated at runtime and excluded from version control via `.gitignore`:

```text
traces/         # JSON execution traces, one per run
memory_db/      # Persistent ChromaDB storage
```
> **Note**
>
> The repository currently includes both `pyproject.toml` and `requirements.txt`.
> Ensure they remain synchronized whenever dependencies are updated.

---

## Prerequisites

- `Python 3.13.2+`
- [Ollama](https://ollama.com/) installed and running locally
- A pulled local model — the CLI currently defaults to `llama3.2:3b`:

```bash
ollama pull llama3.2:3b
```

- Approximately 4-8GB of free RAM (see Known Limitations for memory tips)

---

## Installation

```bash
# Clone repository
git clone <your-repository-url>
cd aeroagent

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download model
ollama pull llama3.2:3b
```

---

## Usage

Run the agent from the command line with any natural-language prompt:

```bash
python main.py "What are the new features in Python 3.13?"
```
### What happens

- The agent prints its reasoning and tool-use steps live to the console (via `rich`), including raw LLM output for each decision.
- It always checks memory first, only searches the web if memory comes up empty, and never searches the web more than once per prompt.
- On completion, it prints a formatted **Final Answer** panel.
- If it fails to reach a final answer, it prints the execution steps so you can see where it got stuck.
- Every run is saved as a JSON trace under `traces/` for later debugging.

Tools available to the agent:

| Tool | Callable by LLM? | Purpose |
|---|---|---|
| `search_memory` | Yes (forced first) | Checks ChromaDB for relevant past interactions |
| `web_search` | Yes (max once per prompt) | DuckDuckGo search, only if memory has no hit |
| `save_to_memory` | No, auto-invoked | Automatically persists `web_search` results; the LLM never decides to call this itself |

Example invocation:

```bash
python main.py "Summarize the latest news on local LLM quantization"
```

---

## Running with Docker

The full stack (app + Ollama) can run in containers via Docker Compose — no local Python environment or Ollama install needed.

### First-time setup

```bash
# Start Ollama first and pull the model into the container
docker compose up -d ollama
docker exec aeroagent-ollama ollama pull llama3.2:3b

# Build and start the full stack
docker compose up -d --build
```

### Usage

Once running, the agent is available as a REST API on `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/health

# Run the agent
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

Interactive API docs (via FastAPI's auto-generated Swagger UI): `http://localhost:8000/docs`

The `/run` endpoint is rate-limited to 10 requests/minute per client IP.

Traces are written to `./traces` on the host (bind-mounted); memory persists in a named Docker volume across restarts.

### Stopping

```bash
docker compose down          # stop containers, keep volumes (models, memory)
docker compose down -v       # stop and wipe volumes too
```

--- 

## Running Tests
Run tests:

```bash
pytest
```

The test suite uses `pytest-asyncio` and mocks `LLMClient.chat_completion`, so it runs fully offline without needing Ollama up, split across four files:

- `test_agent.py` — agent initialization, tool routing, and defensive parsing of plain-text (non-JSON) LLM output.
- `test_agent_loop.py` — agent loop control flow.
- `test_memory.py` — memory stack behavior.
- `test_evals.py` - EvalCase Validation and LLM-as-a-judge scoring logic.

Lint with:

```bash
ruff check .
```

---

## Continuous Integration

Every push and pull request to `main` runs through GitHub Actions (`.github/workflows/ci.yml`):

1. **Lint** — `ruff check .` and `ruff format --check .`
2. **Test** — full `pytest` suite (runs after lint passes)
3. **Docker build** — builds the app image and runs a basic import sanity check (runs after tests pass), using GitHub Actions' cache backend to keep rebuilds fast despite heavy dependencies like `torch` and `chromadb`

The Docker build step does not push the image anywhere yet — it only verifies the image builds and starts correctly.
---

## Roadmap

| Phase | Name | Focus & Deliverables | Status |
|---|---|---|---|
| 1 | **Core Loop and Tooling** | Async state machine, Pydantic validation, Ollama client, DuckDuckGo search tool, local JSON tracing, pytest mocking | Complete |
| 2 | **Memory and Context** | Modular embeddings (sentence-transformers), persistent vector storage (ChromaDB), `save_to_memory` and `search_memory tools` | Complete |
| 3 | **Evaluation and Observability** | LLM-as-a-judge evals, latency/cost tracking, structured logging, prompt versioning | Complete |
| 4 | **Productionization** | FastAPI wrapper, Docker Compose (App + Ollama), GitHub Actions CI/CD (lint, test, Docker build), rate limiting | Complete |
| 5 | **Domain Specialization (optional)** | Finance data tool (e.g. Yahoo Finance API) or a simple recommender microservice | Future |

### End goal

The long-term objective is a production-ready AI agent featuring:

- Asynchronous architecture
- Persistent local memory
- FastAPI REST interface
- Docker deployment
- CI/CD pipeline

---

## Known Limitations and Mitigations

| Limitation | Why it happens | Mitigation |
|---|---|---|
| Small local LLMs produce invalid JSON | Local 1.5B-3B models can forget keys or return plain text instead of structured JSON | Defensive parsing: missing keys are inferred, strings are auto-cast to dicts, and the agent falls back to plain-text extraction if JSON parsing fails |
| Limited RAM | Running an LLM, embedding model, and vector DB together can cause memory pressure / OOM on machines with 8GB RAM | Lightweight stack (small local model plus `all-MiniLM-L6-v2` embeddings, ~80MB) and `OLLAMA_KEEP_ALIVE=5m` to unload the model between uses |
| Tool execution hangs | Network calls (e.g. web search) can time out or get rate-limited | Tools run in `asyncio.to_thread` with explicit `httpx` timeouts and retry logic in the tool registry |
| Vector search noise | Local embedding models are less nuanced than large hosted embeddings | Metadata is stored alongside vectors, and results are returned with explicit relevance scores for the LLM to weigh |
| Stale memory blocks fresh lookups | The agent hard-blocks `web_search` whenever `search_memory` returns any hit, regardless of how old or time-sensitive the cached result is | Caught via the Phase 3 eval suite (`current_events_001`); currently tracked as a known gap rather than fixed, since it requires memory entries to carry freshness/recency metadata - candidate for a future phase |

> **Note**
>
> The model currently configured in `main.py` is `llama3.2:3b`. Earlier planning documents referenced `qwen2.5:1.5b`; update this README if the default model changes.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.