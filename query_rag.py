"""
Query a local RAG index -- retrieve the most relevant chunks for a question.
100% local embeddings via Ollama (nomic-embed-text) + ChromaDB vector store.

Usage:
    python query_rag.py <collection> "<question>"
"""
import sys
import requests
import chromadb

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./chroma_db"


def embed(text):
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def main():
    if len(sys.argv) < 3:
        print('Usage: python query_rag.py <collection> "<question>"')
        sys.exit(1)
    collection = sys.argv[1]
    question = sys.argv[2]

    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(name=f"collection_{collection}")

    vec = embed(question)
    results = coll.query(query_embeddings=[vec], n_results=3)

    print(f"Query: {question}\nCollection: {collection}\n")
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    if not docs:
        print("No results (index empty or collection not populated yet).")
        return
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        print(f"--- Result {i+1} (distance={dist:.4f}, source={meta['source']}) ---")
        print(doc[:500])
        print()


if __name__ == "__main__":
    main()
