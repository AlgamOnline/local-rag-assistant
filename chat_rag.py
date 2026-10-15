"""
Chat with a local LLM (qwen2.5-coder via Ollama), augmented with RAG context
retrieved from your own local knowledge base. No cloud API calls.

Usage:
    python chat_rag.py <collection> "<question>"
"""
import sys
import requests
import chromadb

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5-coder:7b"
DB_PATH = "./chroma_db"
TOP_K = 4


def embed(text):
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve_context(collection, question, top_k=TOP_K):
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(name=f"collection_{collection}")
    vec = embed(question)
    results = coll.query(query_embeddings=[vec], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def build_prompt(collection, question, context_items):
    if not context_items:
        context_block = "(no relevant context found in the local knowledge base)"
    else:
        parts = []
        for i, (doc, meta) in enumerate(context_items):
            parts.append(f"[Context {i+1} - source: {meta['source']}]\n{doc}")
        context_block = "\n\n".join(parts)

    system = (
        f"You are a local AI assistant for the '{collection}' knowledge base. "
        "Answer using the provided context when relevant. "
        "If the context is not sufficient to answer, say so honestly instead "
        "of making things up."
    )
    user = f"CONTEXT FROM KNOWLEDGE BASE:\n{context_block}\n\nQUESTION:\n{question}"
    return system, user


def chat(system, user):
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    if len(sys.argv) < 3:
        print('Usage: python chat_rag.py <collection> "<question>"')
        sys.exit(1)
    collection = sys.argv[1]
    question = sys.argv[2]

    print(f"[RAG] Searching relevant context in collection '{collection}'...")
    context_items = retrieve_context(collection, question)
    print(f"[RAG] Found {len(context_items)} context chunk(s).")

    system, user = build_prompt(collection, question, context_items)

    print("[LLM] Generating answer with qwen2.5-coder:7b...\n")
    answer = chat(system, user)
    print("=== LOCAL AI ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
