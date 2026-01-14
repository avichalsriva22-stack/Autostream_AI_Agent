# Environment setup

Minimum requirements: Python 3.9+

PowerShell (Windows) - recommended (native venv):

1. Open PowerShell in the project root (`d:\Autostream_Agent`).
2. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your PowerShell execution policy blocks activation, run as admin and enable running scripts for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Conda alternative:

```powershell
conda create -n autostream python=3.10 -y
conda activate autostream
pip install -r requirements.txt
```

Quick automated script (Windows PowerShell): run `.\setup_env.ps1` (see script in repo).

Notes:
- The `requirements.txt` installs both OpenAI and Google Generative AI integrations (`langchain-openai`, `langchain-google-genai`, `google-generativeai`). Set the credentials you actually plan to use.  
- Vector-store back ends include both `faiss-cpu` and `chromadb`. Feel free to remove one if you prefer a lighter setup.  
- `langgraph-checkpoint-sqlite` and `sqlite-vec` are included so the LangGraph agent can store checkpoints in SQLite.

## Agent graph usage

The conversational agent is assembled in `graph.py`. Key points:

1. **LLM selection**: `select_llm()` prefers the provider specified via `LLM_PROVIDER` (`openai` or `google`). If that is unset it auto-detects based on the available API keys (`OPENAI_API_KEY` / `GOOGLE_API_KEY`).
2. **Sqlite-backed memory**: `build_agent()` compiles a LangGraph `StateGraph` with `SqliteSaver`. Pass a filesystem path for persistent memory (default: `checkpoints/agent.sqlite`). Use `":memory:"` to keep checkpoints in RAM.
3. **Invoking the agent**: when calling the compiled app, include a `config` with a `thread_id` so checkpoints can be correlated, e.g.

```python
# Autostream Agent

This repository contains a small LangGraph/AutoGen-based conversational agent with a FAISS-backed knowledge store and SQLite checkpoints. The sections below explain how to run the project locally, the architecture rationale, and how you could integrate the agent with WhatsApp via webhooks.

## How to run the project locally

Prerequisites: Python 3.9+, Git, and either PowerShell (Windows) or a POSIX shell.

1) Clone and create a virtual environment (Windows PowerShell):

```powershell
cd D:\Autostream_Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If using PowerShell execution policy restrictions, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

2) (Optional) Build the vector store / knowledge index:

```powershell
python ingest.py
```

3) Start the interactive demo (simple REPL):

```powershell
python main.py
```

During the REPL you can send messages until you type `exit` or `quit`.

Notes:
- If you prefer Conda, create and activate an environment then install `requirements.txt`.
- Environment variables control LLM providers/keys (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LLM_PROVIDER`).
- The repository includes example scripts and `setup_env.ps1` to help with Windows setup.

## Architecture Explanation 

This agent composes a small pipeline built around LangGraph (for stateful graph-based orchestration) and AutoGen-style patterns for modular tool and LLM interactions. LangGraph provides an explicit state graph and checkpoint/saver primitives, which makes reasoning about conversational flows and long-running sessions straightforward: each node represents a step (question, tool call, validation), and the graph enforces ordering and state transitions. AutoGen-style components simplify writing small, testable tools (e.g., knowledge retrieval, lead-capture) and let the LLM orchestrate high-level decisions while guarded logic enforces safety.

State management uses a combination of in-process `ConversationState` objects and durable checkpointing via SQLite (configured via the `SqliteSaver`/checkpoint layer). Each user session/thread has a `thread_id` that keys into persistent checkpoints; intermediate states (messages, user profile fields, tool outputs) are serialized periodically so the agent can recover from restarts or continue multi-step interactions. The vector store (FAISS) is used for retrieval-augmented responses; it is kept separate from transient conversation state so RAG lookups remain immutable while conversation context evolves. This separation reduces corruption risk and simplifies scaling and backups.

Why LangGraph / AutoGen: the combination provides structured, auditable conversational flows (LangGraph) and modular tool abstractions (AutoGen) that together make it easier to reason about when external actions are taken, persist checkpoints, and test interactions end-to-end.

## WhatsApp Deployment — Webhook Integration (overview)

To integrate this agent with WhatsApp, you typically use a WhatsApp Business API provider (e.g., Meta's WhatsApp Business API, Twilio, or other BSP). The flow below describes a webhook-based integration:

1) Choose a provider and set up a WhatsApp number and credentials. For development Twilio's WhatsApp Sandbox is convenient.

2) Host the agent and expose a webhook endpoint (HTTPS). The agent app must run a small HTTP server that accepts POST webhooks from WhatsApp provider. Example endpoints:

- `POST /webhook/whatsapp` — receives incoming messages
- `POST /webhook/whatsapp/status` — optional delivery receipts

3) Incoming message handling:
- Verify provider signatures (e.g., Twilio X-Twilio-Signature or Meta X-Hub-Signature) to ensure authenticity.
- Parse the webhook JSON to extract sender id (phone), message text, and any media.
- Map provider sender id to an internal `thread_id` (one thread per phone number) and build a `ConversationState` with prior messages loaded from checkpoints.

4) Invoke the agent:
- Call `agent.invoke(state, config={"configurable": {"thread_id": "whatsapp:<phone>"}})` or equivalent wrapper.
- Collect the agent's reply(s) and any requested actions (e.g., send media or call a tool).

5) Reply to the user via provider API:
- Use the provider's REST API to send messages back to the WhatsApp number. For Twilio, POST to Twilio Messages API with the `to` number and message body; for Meta's API, POST to `/v1/messages` with `recipient` and `message` fields.

6) Session & state considerations:
- Persist checkpoints after each agent turn so restarts don't lose context.
- Enforce idempotency (webhook retries can occur): deduplicate by message id.
- Handle media separately: download provider-hosted media to temp storage, then pass a reference to the agent if needed.

7) Security and scaling:
- Terminate TLS at the host or reverse proxy (e.g., nginx, Cloud Run). Use provider signature verification for webhooks.
- Use a message queue (e.g., Redis or SQS) to buffer incoming webhooks and allow horizontal scaling of the agent workers.

Implementation notes and tips:
- For prototyping, run the agent behind a tunnel (ngrok) and register the forwarded HTTPS URL in Twilio or Meta webhook settings.
- Keep mapping between WhatsApp phone numbers and `thread_id` deterministic and revocable (e.g., `whatsapp:+1234567890`) to simplify session lookup.
- Respect provider rules for message templates (required for outbound notifications when not replying to recent incoming messages).

## Where to look in this repo

- Agent graph/config: [graph.py](graph.py)
- App entrypoints: [main.py](main.py), [ingest.py](ingest.py)
- Node/tool definitions: [nodes.py](nodes.py)
- State types and persistence: [state.py](state.py)



