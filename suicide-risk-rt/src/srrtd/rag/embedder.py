from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str
    batch_size: int = 64


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbedderConfig):
        from sentence_transformers import SentenceTransformer

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        self.cfg = cfg
        # Prefer GPU if available. Some older versions may not accept `device=`.
        try:
            self.model = SentenceTransformer(cfg.model_name, device=device)
        except TypeError:
            self.model = SentenceTransformer(cfg.model_name)
            try:
                self.model.to(device)
            except Exception:
                pass

    @property
    def dim(self) -> int:
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # fallback common dims
            return 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode(
            texts,
            batch_size=int(self.cfg.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype("float32").tolist()
