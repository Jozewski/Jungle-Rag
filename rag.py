"""
RAG Engine — loads knowledge base files, chunks them, embeds with
sentence-transformers, stores in ChromaDB, and retrieves top-k chunks.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions

KB_DIR = Path(__file__).parent / "knowledge_base"
COLLECTION_NAME = "survival_guide"
CHUNK_SIZE = 400        # target words per chunk
CHUNK_OVERLAP = 60      # word overlap between chunks
TOP_K = 5              # number of chunks to retrieve


# ---------------------------------------------------------------------------
# Embedding function (sentence-transformers, runs locally, free)
# ---------------------------------------------------------------------------
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> List[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_idx": chunk_idx,
        })
        chunk_idx += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_file(path: Path) -> str:
    """Load a knowledge base file into a string."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".json":
        try:
            data = json.loads(text)
            # Flatten JSON into readable text
            return _flatten_json(data)
        except json.JSONDecodeError:
            return text
    # .md and .txt — return as-is
    return text


def _flatten_json(obj, depth=0) -> str:
    """Recursively flatten a JSON object to readable plain text."""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{'  ' * depth}{k}:")
                lines.append(_flatten_json(v, depth + 1))
            else:
                lines.append(f"{'  ' * depth}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            lines.append(_flatten_json(item, depth))
    else:
        lines.append(str(obj))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

def _get_client():
    """Return a persistent ChromaDB client."""
    persist_dir = Path(__file__).parent / ".chromadb"
    persist_dir.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Load all knowledge base files, chunk them, and store in ChromaDB.
    Skips rebuild if the collection already exists (unless force_rebuild=True).
    """
    client = _get_client()

    # Check if collection already populated
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedding_fn,
        )
        if collection.count() > 0:
            print(f"[RAG] Using existing index ({collection.count()} chunks).")
            return collection
        client.delete_collection(COLLECTION_NAME)

    if COLLECTION_NAME in existing and force_rebuild:
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for path in sorted(KB_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        print(f"[RAG] Indexing: {path.name}")
        raw = _load_file(path)
        chunks = _chunk_text(raw, source=path.name)
        all_chunks.extend(chunks)

    # Batch-add to ChromaDB
    BATCH = 100
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start : start + BATCH]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in batch],
            ids=[f"{c['source']}_{c['chunk_idx']}" for c in batch],
        )

    print(f"[RAG] Indexed {len(all_chunks)} chunks from {KB_DIR}.")
    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns a list of dicts with keys: text, source, score.
    """
    client = _get_client()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": round(1 - dist, 4),  # cosine similarity
        })

    return chunks
