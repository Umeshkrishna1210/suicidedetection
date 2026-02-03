from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ChromaConfig:
    chroma_dir: str
    collection: str


def get_client(chroma_dir: str):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir))


def get_or_create_collection(client, name: str):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def reset_collection(client, name: str) -> None:
    try:
        client.delete_collection(name=name)
    except Exception:
        pass


def add_documents(
    collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    # Chroma supports batched inserts
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
