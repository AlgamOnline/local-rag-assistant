# Local RAG Assistant 🦙

A fully **local, private, zero-API-cost** Retrieval-Augmented Generation (RAG)
system — built with [Ollama](https://ollama.com) + [ChromaDB](https://www.trychroma.com/),
no OpenAI/cloud API keys required.

## Why local?

Most RAG tutorials assume you'll pay per-token for embeddings and completions
via OpenAI/Anthropic APIs. This project proves you can run a complete
RAG pipeline — embeddings, vector search, and LLM chat — **entirely on
consumer hardware** (tested on an RTX 4060 8GB VRAM), with zero per-query cost
and zero data leaving your machine.

Useful for:
- Personal knowledge bases over private notes/docs you don't want sent to a cloud API
- Prototyping RAG architecture before committing to a paid API
- Offline-first tools, air-gapped environments, or just avoiding vendor lock-in

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Documents  │────▶│  nomic-embed-text │────▶│    ChromaDB      │
│ (md/txt/py) │     │  (via Ollama)      │     │ (local vectors)  │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │  qwen2.5-coder:7b │◀────│  Top-K retrieval │
                     │  (via Ollama)      │     │  (query time)    │
                     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                     Answer grounded in your
                     own documents, generated
                     100% locally.
```

## Stack

- **LLM**: `qwen2.5-coder:7b` (Ollama) — chat/reasoning
- **Embeddings**: `nomic-embed-text` (Ollama) — vector embeddings
- **Vector DB**: [ChromaDB](https://www.trychroma.com/) — local persistent store
- **Language**: Python 3.11

## Setup

1. Install [Ollama](https://ollama.com/download) and pull the models:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ollama serve
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Put your own documents in `sample_data/` (or edit `SOURCES` in
   `index_project.py` to point at your own folders), then build the index:
   ```bash
   python index_project.py
   ```

4. Query the index directly (no LLM, just retrieval):
   ```bash
   python query_rag.py notes "what did I write about X?"
   ```

5. Or chat with the local LLM, grounded in your documents:
   ```bash
   python chat_rag.py notes "Summarize my notes about X"
   ```

## Files

| File | Purpose |
|---|---|
| `index_project.py` | Scans folders, chunks text, embeds via Ollama, stores in ChromaDB |
| `query_rag.py` | Raw retrieval — returns the top-K most relevant chunks for a query |
| `chat_rag.py` | Full RAG pipeline — retrieves context, then asks the local LLM to answer |
| `log_work.py` | Logs structured work sessions (problem → approach → result) as a dataset foundation for future fine-tuning experiments |

## Notes on hardware

Tested end-to-end on: Intel i5-12400F, 32GB RAM, RTX 4060 8GB VRAM.
A 7B model + RAG retrieval runs comfortably; response latency for a
context-heavy query is roughly 30s–2min depending on context size —
slower than a cloud API, but free and fully private.

## License

MIT — use freely.
