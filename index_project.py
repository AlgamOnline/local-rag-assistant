"""
Local RAG Indexer
Scans text/code files per knowledge "collection", embeds them via a local
Ollama model (nomic-embed-text), and stores vectors in a local ChromaDB.

100% local: no OpenAI/cloud API calls, no data leaves your machine.
Useful for building a private knowledge base over personal notes, docs,
or project files without paying per-token for embeddings.

Usage:
    python index_project.py

Configure your own collections in SOURCES below.
"""
import os
import hashlib
import requests
import chromadb

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./chroma_db"

# collection_name -> list of folders to scan
SOURCES = {
    "notes": [
        "./sample_data/notes",
    ],
    "docs": [
        "./sample_data/docs",
    ],
}

TEXT_EXT = {".md", ".txt", ".json", ".py", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml"}
MAX_CHARS = 2000  # chunk size in characters
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "chroma_db"}


def chunk_text(text, size=MAX_CHARS):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def embed(text):
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def walk_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in TEXT_EXT:
                yield os.path.join(dirpath, fn)


def index_collection(name, folders, client):
    coll = client.get_or_create_collection(name=f"collection_{name}")
    total_chunks = 0
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"  [skip] folder not found: {folder}")
            continue
        for filepath in walk_files(folder):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"  [error read] {filepath}: {e}")
                continue
            if not content.strip():
                continue
            for idx, chunk in enumerate(chunk_text(content)):
                if not chunk.strip():
                    continue
                doc_id = hashlib.md5(f"{filepath}:{idx}".encode()).hexdigest()
                try:
                    vec = embed(chunk)
                except Exception as e:
                    print(f"  [error embed] {filepath} chunk {idx}: {e}")
                    continue
                coll.upsert(
                    ids=[doc_id],
                    embeddings=[vec],
                    documents=[chunk],
                    metadatas=[{"source": filepath, "chunk": idx}],
                )
                total_chunks += 1
    print(f"[{name}] total chunks indexed: {total_chunks}")
    return total_chunks


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    grand_total = 0
    for name, folders in SOURCES.items():
        print(f"\n=== Indexing collection: {name} ===")
        grand_total += index_collection(name, folders, client)
    print(f"\nTOTAL chunks indexed across all collections: {grand_total}")


if __name__ == "__main__":
    main()
