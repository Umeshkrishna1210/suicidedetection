from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from srrtd.rag.chroma_store import get_client, get_or_create_collection
from srrtd.rag.embedder import EmbedderConfig, SentenceTransformerEmbedder


@dataclass(frozen=True)
class RagConfig:
    embed_model: str
    embed_dim: int
    top_k: int
    chroma_dir: str
    collection: str


class RagRetriever:
    def __init__(self, cfg: RagConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root
        chroma_path = (project_root / cfg.chroma_dir).resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = get_client(str(chroma_path))
        self.collection = get_or_create_collection(self.client, cfg.collection)
        self.embedder = SentenceTransformerEmbedder(EmbedderConfig(model_name=cfg.embed_model))

    def retrieve_context_vec(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, int(self.cfg.embed_dim)), device=device)

        try:
            if hasattr(self.collection, "count") and int(self.collection.count()) == 0:
                return torch.zeros((len(texts), int(self.cfg.embed_dim)), device=device)
        except Exception:
            pass

        q_emb = self.embedder.encode(texts)
        try:
            res = self.collection.query(
                query_embeddings=q_emb,
                n_results=int(self.cfg.top_k),
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        except Exception:
            return torch.zeros((len(texts), int(self.cfg.embed_dim)), device=device)

        ctx_vecs: list[np.ndarray] = []
        # res["embeddings"]: list[batch][k][dim]
        embeddings = res.get("embeddings", None)
        for i in range(len(texts)):
            if embeddings is None or i >= len(embeddings) or embeddings[i] is None:
                ctx_vecs.append(np.zeros((int(self.cfg.embed_dim),), dtype=np.float32))
                continue

            arr = np.asarray(embeddings[i], dtype=np.float32)
            if arr.size == 0:
                ctx_vecs.append(np.zeros((int(self.cfg.embed_dim),), dtype=np.float32))
                continue

            # Allow either a single embedding (dim,) or a set (k, dim)
            if arr.ndim == 1:
                ctx_vecs.append(arr)
            else:
                arr2 = arr.reshape((-1, arr.shape[-1]))
                ctx_vecs.append(arr2.mean(axis=0))

        ctx = np.stack(ctx_vecs, axis=0)
        return torch.tensor(ctx, dtype=torch.float32, device=device)

    def retrieve_texts(self, text: str) -> list[str]:
        q_emb = self.embedder.encode([text])
        res = self.collection.query(query_embeddings=q_emb, n_results=int(self.cfg.top_k), include=["documents"])
        docs = res.get("documents", [[]])[0] or []
        return [str(d) for d in docs]
