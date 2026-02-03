from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srrtd.data.schema import DatasetBundle, PostRecord
from srrtd.rag.chroma_store import add_documents, get_client, get_or_create_collection, reset_collection
from srrtd.rag.embedder import EmbedderConfig, SentenceTransformerEmbedder


@dataclass(frozen=True)
class RagIndexConfig:
    embed_model: str
    chroma_dir: str
    collection: str


def build_rag_index(cfg: RagIndexConfig, project_root: Path, bundle: DatasetBundle, reset: bool = True) -> dict[str, Any]:
    chroma_path = (project_root / cfg.chroma_dir).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = get_client(str(chroma_path))
    if reset:
        reset_collection(client, cfg.collection)
    collection = get_or_create_collection(client, cfg.collection)

    embedder = SentenceTransformerEmbedder(EmbedderConfig(model_name=cfg.embed_model))

    records: list[PostRecord] = []
    records.extend(bundle.train.records)
    records.extend(bundle.val.records)
    records.extend(bundle.test.records)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []

    for i, r in enumerate(records):
        ids.append(f"{r.user_id}:{int(r.timestamp)}:{i}")
        docs.append(r.text)
        metas.append({"user_id": r.user_id, "timestamp": int(r.timestamp), "risk": int(r.risk), "emotion": int(r.emotion), "lang": r.lang})

    # Batch embed + add
    bs = 256
    for start in range(0, len(docs), bs):
        end = min(len(docs), start + bs)
        emb = embedder.encode(docs[start:end])
        add_documents(collection, ids[start:end], docs[start:end], emb, metas[start:end])

    return {"count": len(docs), "embed_dim": int(embedder.dim), "collection": cfg.collection, "chroma_dir": str(chroma_path)}
