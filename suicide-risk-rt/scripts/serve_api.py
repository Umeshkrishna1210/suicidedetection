from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from fastapi import Query
from pydantic import BaseModel, Field
try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    from transformers.models.auto.tokenization_auto import AutoTokenizer

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle, load_multitask_bundles
from srrtd.models.community import CommunityGraph, CommunityGnnContext
from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.rag.retriever import RagConfig, RagRetriever
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.device import resolve_device
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


def _telugu_chars(s: str) -> int:
    return sum(1 for c in s if 0x0C00 <= ord(c) <= 0x0C7F)


def _maybe_fix_mojibake(text: str) -> str:
    """Best-effort recovery for common UTF-8-as-Latin1 mojibake.

    PowerShell/Windows clients sometimes send UTF-8 bytes but they end up decoded
    as Latin-1, producing strings like "à°..." for Telugu.
    """

    if not text:
        return text

    # If text already contains Telugu script, leave it.
    if _telugu_chars(text) > 0:
        return text

    # Heuristic: mojibake often contains these sequences.
    if ("à" not in text) and ("Ã" not in text) and ("Â" not in text):
        return text

    try:
        fixed = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text

    # Accept the fix only if it produced non-trivial Telugu script.
    if _telugu_chars(fixed) > 0:
        return fixed
    return text


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str = Field(default="anon")
    update_memory: bool = Field(default=True)


class PredictResponse(BaseModel):
    user_id: str
    text: str
    risk_label: str
    risk_prob_suicide: float
    risk_probs: dict[str, float]
    emotion_top: str
    emotion_probs: dict[str, float]
    emotion_confidence_pct: dict[str, float]
    module_debug: dict[str, Any] | None = None
    disclaimer: str


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--ckpt", required=True, help="Path to a .pt checkpoint (e.g., outputs/<run>/best.pt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    ckpt_path = (_ROOT / str(args.ckpt)).resolve() if not Path(str(args.ckpt)).is_absolute() else Path(str(args.ckpt))
    ckpt = _load_ckpt(ckpt_path)

    model_cfg = model_config_from_yaml(cfg)
    # Align head sizes to checkpoint label spaces.
    model_cfg = model_cfg.__class__(
        **{
            **model_cfg.__dict__,
            "risk_classes": list(ckpt.get("risk_classes", model_cfg.risk_classes)),
            "emotion_classes": list(ckpt.get("emotion_classes", model_cfg.emotion_classes)),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt.get("tokenizer_name", model_cfg.encoder_name)))

    model = SrrtdMultiTaskModel(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Optional RAG
    rag = None
    rag_count = None
    if bool(model_cfg.use_rag):
        rag_cfg_d = dict(cfg.get("model", {}).get("rag", {}) or {})
        rag_cfg = RagConfig(
            embed_model=str(rag_cfg_d.get("embed_model")),
            embed_dim=int(rag_cfg_d.get("embed_dim", model_cfg.rag_embed_dim)),
            top_k=int(rag_cfg_d.get("top_k", 5)),
            chroma_dir=str(rag_cfg_d.get("chroma_dir")),
            collection=str(rag_cfg_d.get("collection")),
        )
        rag = RagRetriever(rag_cfg, _ROOT)
        try:
            if hasattr(rag.collection, "count"):
                rag_count = int(rag.collection.count())
        except Exception:
            rag_count = None

    # Optional GNN
    gnn_ctx = None
    graph = None
    if bool(model_cfg.use_gnn):
        desired_max_neighbors = int(cfg.get("model", {}).get("gnn", {}).get("max_neighbors", 16))
        graph_json = ckpt_path.parent / "community_graph.json"
        if graph_json.exists():
            try:
                import json

                with open(graph_json, "r", encoding="utf-8") as f:
                    g = json.load(f)
                if isinstance(g, dict) and isinstance(g.get("adj"), dict):
                    graph = CommunityGraph()
                    graph.adj = {str(k): [str(x) for x in (v or [])] for k, v in (g.get("adj") or {}).items()}

                    try:
                        max_deg = max((len(v) for v in graph.adj.values()), default=0)
                    except Exception:
                        max_deg = 0
                    meta_max = None
                    try:
                        meta = g.get("meta") if isinstance(g.get("meta"), dict) else None
                        if meta and "max_neighbors" in meta:
                            meta_max = int(meta.get("max_neighbors"))
                    except Exception:
                        meta_max = None
                    effective_saved_cap = meta_max if meta_max is not None else max_deg
                    if desired_max_neighbors > int(effective_saved_cap):
                        graph = None
            except Exception:
                graph = None

        if graph is None:
            mode = str(cfg.get("data", {}).get("mode", "single")).lower()
            try:
                if mode == "multitask":
                    rb, eb = load_multitask_bundles(cfg, _ROOT)
                    recs = [{"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp} for s in (rb.train, rb.val, rb.test) for r in s.records]
                    recs += [{"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp} for s in (eb.train, eb.val, eb.test) for r in s.records]
                else:
                    b = load_dataset_bundle(cfg, _ROOT)
                    recs = [{"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp} for s in (b.train, b.val, b.test) for r in s.records]
                graph = CommunityGraph.build_from_records(
                    records=recs,
                    seed=int(cfg.get("project", {}).get("seed", 42)),
                    max_neighbors=desired_max_neighbors,
                )
            except DatasetConfirmationRequired:
                graph = None

        fused_dim = int(model.encoder.config.hidden_size)
        if model_cfg.use_memory:
            fused_dim += int(model_cfg.memory.user_state_dim)
        gnn_ctx = CommunityGnnContext(
            user_state_dim=int(model_cfg.memory.user_state_dim),
            fused_dim=fused_dim,
            gnn_cfg=model_cfg.gnn,
        ).to(device)
        gnn_sd = ckpt.get("gnn_state_dict") if isinstance(ckpt, dict) else None
        if isinstance(gnn_sd, dict):
            gnn_ctx.load_state_dict(gnn_sd, strict=False)
            gnn_ctx.eval()
        else:
            gnn_ctx = None
            graph = None

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    threshold_high = float(cfg.get("inference", {}).get("threshold_high", 0.55))

    app = FastAPI(title="SRRTD API", version="0.1")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "device": str(device),
            "encoder": str(model_cfg.encoder_name),
            "risk_classes": list(model_cfg.risk_classes),
            "emotion_classes": list(model_cfg.emotion_classes),
            "use_memory": bool(model_cfg.use_memory),
            "use_rag": bool(model_cfg.use_rag),
            "rag_index_count": rag_count,
            "use_gnn": bool(model_cfg.use_gnn),
            "gnn_loaded": bool(gnn_ctx is not None and graph is not None),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        req: PredictRequest,
        debug: bool = Query(default=False),
        use_rag: bool = Query(default=True),
        use_gnn: bool = Query(default=True),
    ):
        text_raw = _maybe_fix_mojibake(req.text)
        text = privacy_preprocess(text_raw, privacy_cfg)
        user_id = req.user_id or "anon"

        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        rag_ctx = None
        rag_texts: list[str] = []
        if rag is not None and bool(use_rag):
            rag_ctx = rag.retrieve_context_vec([text], device=device)
            if debug:
                try:
                    rag_texts = rag.retrieve_texts(text)
                except Exception:
                    rag_texts = []

        gnn_add = None
        neigh_ids: list[str] = []
        if graph is not None:
            try:
                neigh_ids = list(graph.neighbors(str(user_id)))
            except Exception:
                neigh_ids = []
        if gnn_ctx is not None and graph is not None and model.memory_store is not None and bool(use_gnn):
            gnn_add = gnn_ctx(
                user_ids=[user_id],
                device=device,
                graph=graph,
                user_state_lookup=lambda uid, dev, dim: model.memory_store.get_long(uid, dev, dim),
            )

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=[user_id],
                update_memory=bool(req.update_memory),
                gnn_context=gnn_add,
                rag_context=rag_ctx,
            )

        risk_probs_t = torch.softmax(out["risk_logits"], dim=-1)[0].detach().cpu()
        emo_probs_t = torch.softmax(out["emotion_logits"], dim=-1)[0].detach().cpu()

        risk_classes = list(model_cfg.risk_classes)
        emo_classes = list(model_cfg.emotion_classes)

        risk_probs = {c: float(risk_probs_t[i].item()) for i, c in enumerate(risk_classes)}
        emo_probs = {c: float(emo_probs_t[i].item()) for i, c in enumerate(emo_classes)}
        emo_conf_pct = {c: float(emo_probs_t[i].item() * 100.0) for i, c in enumerate(emo_classes)}

        # Locate the "suicide" class index robustly (don't assume ordering).
        suicide_idx = 1 if len(risk_classes) > 1 else 0
        try:
            classes_norm = [str(c).strip().lower() for c in (risk_classes or [])]
            if "suicide" in classes_norm:
                suicide_idx = int(classes_norm.index("suicide"))
        except Exception:
            pass
        p_suicide = float(risk_probs_t[suicide_idx].item())
        risk_label = "high" if p_suicide >= threshold_high else "low"

        top_emo_idx = int(torch.argmax(emo_probs_t).item())
        emotion_top = emo_classes[top_emo_idx] if 0 <= top_emo_idx < len(emo_classes) else "unknown"

        module_debug = None
        if debug:
            module_debug = {
                "device": str(device),
                "rag_index_count": rag_count,
                "rag_ctx_norm": (float(torch.linalg.vector_norm(rag_ctx[0]).item()) if rag_ctx is not None else None),
                "gnn_ctx_norm": (float(torch.linalg.vector_norm(gnn_add[0]).item()) if gnn_add is not None else None),
                "gnn_neighbor_count": int(len(neigh_ids)),
                "gnn_neighbor_sample": [str(x) for x in (neigh_ids or [])[:16]],
                "retrieved_texts": [str(t)[:300] for t in (rag_texts or [])[:5]],
            }

            if model.memory_store is not None and model.cfg.use_memory:
                try:
                    long_state = model.memory_store.get_long(
                        user_id,
                        device=device,
                        dim=int(model.cfg.memory.user_state_dim),
                    )
                    module_debug["memory_long_norm"] = float(torch.linalg.vector_norm(long_state).item())
                except Exception:
                    module_debug["memory_long_norm"] = None

        return PredictResponse(
            user_id=user_id,
            text=text,
            risk_label=risk_label,
            risk_prob_suicide=p_suicide,
            risk_probs=risk_probs,
            emotion_top=emotion_top,
            emotion_probs=emo_probs,
            emotion_confidence_pct=emo_conf_pct,
            module_debug=module_debug,
            disclaimer=(
                "This output is a research risk/emotion signal, not a diagnosis. "
                "If someone may be in immediate danger, contact local emergency services." 
            ),
        )

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
