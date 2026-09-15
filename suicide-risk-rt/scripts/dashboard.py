from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path
from typing import Any, Tuple
import hashlib

import numpy as np
import streamlit as st
import torch
try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    from transformers.models.auto.tokenization_auto import AutoTokenizer

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle, load_multitask_bundles
from srrtd.models.factory import model_config_from_yaml
from srrtd.models.community import CommunityGraph, CommunityGnnContext
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.rag.retriever import RagConfig, RagRetriever
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.device import resolve_device
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


def _telugu_chars(s: str) -> int:
    # Telugu block: U+0C00..U+0C7F
    try:
        return sum(1 for ch in s if 0x0C00 <= ord(ch) <= 0x0C7F)
    except Exception:
        return 0


def _maybe_fix_mojibake(text: str) -> str:
    """Best-effort recovery for common UTF-8->Latin-1 mojibake.

    This happens when UTF-8 bytes are incorrectly decoded as Latin-1, producing
    sequences like 'à°...' for Telugu input.
    """

    if not isinstance(text, str) or not text:
        return text

    # Fast-path: if it already contains Telugu, do nothing.
    if _telugu_chars(text) >= 2:
        return text

    # Heuristic: mojibake often contains lots of Latin-1 supplement glyphs.
    suspect = sum(1 for ch in text if 0x00A0 <= ord(ch) <= 0x00FF)
    if suspect < 4:
        return text

    try:
        candidate = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text

    # Only accept the fix if it increases Telugu characters.
    if _telugu_chars(candidate) > _telugu_chars(text):
        return candidate
    return text


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_resource
def _load_model(config_path: str, ckpt_path: str, overrides: Tuple[str, ...], config_fingerprint: str):
    cfg = load_yaml(config_path)
    cfg = apply_overrides(cfg, list(overrides) if overrides else None)

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    ckpt_p = (_ROOT / str(ckpt_path)).resolve() if not Path(str(ckpt_path)).is_absolute() else Path(str(ckpt_path))
    ckpt = _load_ckpt(ckpt_p)

    model_cfg = model_config_from_yaml(cfg)
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
        # Prefer the exact graph used during training, if saved.
        ckpt_p2 = (_ROOT / str(ckpt_path)).resolve() if not Path(str(ckpt_path)).is_absolute() else Path(str(ckpt_path))
        graph_json = ckpt_p2.parent / "community_graph.json"
        desired_max_neighbors = int(cfg.get("model", {}).get("gnn", {}).get("max_neighbors", 16))

        if graph_json.exists():
            try:
                import json

                with open(graph_json, "r", encoding="utf-8") as f:
                    g = json.load(f)
                if isinstance(g, dict) and isinstance(g.get("adj"), dict):
                    graph = CommunityGraph()
                    graph.adj = {str(k): [str(x) for x in (v or [])] for k, v in (g.get("adj") or {}).items()}

                    # If the saved graph is clearly capped lower than desired, rebuild from dataset.
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
            # Fallback: build from configured dataset bundles (can be slow on very large datasets).
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

    return cfg, model_cfg, tokenizer, model, device, privacy_cfg, threshold_high, rag, rag_count, gnn_ctx, graph


def _probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()


def _label_risk(risk_probs: np.ndarray, risk_classes: list[str], threshold_high: float) -> tuple[str, float]:
    suicide_idx = 1 if len(risk_classes) > 1 else 0
    try:
        classes_norm = [str(c).strip().lower() for c in (risk_classes or [])]
        if "suicide" in classes_norm:
            suicide_idx = int(classes_norm.index("suicide"))
    except Exception:
        pass
    p_suicide = float(risk_probs[suicide_idx]) if len(risk_probs) else 0.0
    risk_label = "HIGH" if p_suicide >= float(threshold_high) else "LOW"
    return risk_label, p_suicide


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default="configs/local_multitask.yaml")
    ap.add_argument("--ckpt", default="outputs/risk_only/best.pt")
    ap.add_argument("--overrides", nargs="*", default=None)
    args, _unknown = ap.parse_known_args()

    overrides = tuple(args.overrides or [])

    st.set_page_config(page_title="SRRTD Dashboard", layout="wide")

    st.title("Suicide Risk and Emotion Dashboard")
    st.caption("Not a diagnosis. Use for research/demo only.")

    with st.sidebar:
        st.subheader("Suicide Risk and Emotion Detection Model")
        config_path = st.text_input("Config path", value=str(args.config))
        ckpt_path = st.text_input("Checkpoint path", value=str(args.ckpt))

        # Fingerprint config content so Streamlit cache invalidates when the file changes.
        try:
            cfg_bytes = Path(config_path).read_bytes() if Path(config_path).exists() else (_ROOT / config_path).read_bytes()
            config_fingerprint = hashlib.sha1(cfg_bytes).hexdigest()
        except Exception:
            config_fingerprint = ""

        st.subheader("Modules")
        use_rag_now = st.checkbox("Use RAG context", value=True)
        use_gnn_now = st.checkbox("Use GNN context", value=True)
        auto_map_user = st.checkbox("Auto-map user_id for GNN", value=True)
        show_debug = st.checkbox("Show module debug", value=False)

        st.subheader("Session")
        user_id = st.text_input("user_id", value="demo_user")
        update_memory = st.checkbox("Update memory (streaming context)", value=True)
        if st.button("Reset memory state"):
            # Reset happens on the cached model object.
            try:
                _cfg, _mc, _tok, _m, _dev, _pc, _th, _rag, _rc, _gnn, _graph = _load_model(
                    config_path, ckpt_path, overrides, config_fingerprint
                )
                _m.reset_streaming_state()
                st.success("Memory reset")
            except Exception:
                st.warning("Could not reset memory")

    try:
        cfg, model_cfg, tokenizer, model, device, privacy_cfg, threshold_high, rag, rag_count, gnn_ctx, graph = _load_model(
            config_path, ckpt_path, overrides, config_fingerprint
        )
    except Exception as e:
        st.error(f"Failed to load model/checkpoint: {e}")
        st.stop()

    # If the provided user_id isn't present in the community graph, GNN will have no neighbors.
    # For a clean demo (and to make the GNN effect visible), optionally map user_id deterministically
    # onto an existing graph user.
    effective_user_id = user_id
    if auto_map_user and graph is not None and getattr(graph, "adj", None):
        if str(user_id) not in graph.adj:
            keys = sorted(list(graph.adj.keys()))
            if keys:
                h = int(hashlib.sha1(str(user_id).encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
                effective_user_id = keys[h % len(keys)]

    with st.sidebar:
        st.caption(f"Device: {device}")
        if effective_user_id != user_id:
            st.caption(f"GNN effective_user_id: {effective_user_id}")
        if model_cfg.use_rag:
            st.caption(f"RAG enabled in config: true (index count={rag_count if rag_count is not None else 'unknown'})")
        else:
            st.caption("RAG enabled in config: false")
        if model_cfg.use_gnn:
            st.caption(f"GNN enabled in config: {'true' if gnn_ctx is not None else 'true (but missing weights)'}")
        else:
            st.caption("GNN enabled in config: false")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        text = st.text_area("Text", height=180, value="I feel hopeless and I don't know what to do.")
        context_lines = st.text_area(
            "Optional context (one prior post per line, used to update Memory before prediction)",
            height=120,
            value="",
        )
        run = st.button("Predict")

    if run:
        text_raw = _maybe_fix_mojibake(text)
        clean_text = privacy_preprocess(text_raw, privacy_cfg)
        ctx_texts = [
            privacy_preprocess(_maybe_fix_mojibake(t), privacy_cfg)
            for t in (context_lines or "").splitlines()
            if t.strip()
        ]

        enc = tokenizer(
            [clean_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Prepare contexts
        rag_ctx = None
        rag_texts = []
        if rag is not None and bool(use_rag_now):
            rag_ctx = rag.retrieve_context_vec([clean_text], device=device)
            if show_debug:
                try:
                    rag_texts = rag.retrieve_texts(clean_text)
                except Exception:
                    rag_texts = []

        gnn_add = None
        neigh_ids: list[str] = []
        if graph is not None:
            try:
                neigh_ids = list(graph.neighbors(str(effective_user_id)))
            except Exception:
                neigh_ids = []
        if gnn_ctx is not None and graph is not None and model.memory_store is not None and bool(use_gnn_now):
            gnn_add = gnn_ctx(
                user_ids=[effective_user_id],
                device=device,
                graph=graph,
                user_state_lookup=lambda uid, dev, dim: model.memory_store.get_long(uid, dev, dim),
            )

        def _predict(rag_on: bool, gnn_on: bool):
            with torch.no_grad():
                return model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    user_ids=[effective_user_id],
                    update_memory=bool(update_memory),
                    gnn_context=(gnn_add if gnn_on else None),
                    rag_context=(rag_ctx if rag_on else None),
                )

        def _ingest_memory_context():
            if not ctx_texts:
                return
            # Feed prior posts to memory. We intentionally do NOT apply RAG/GNN here,
            # to keep the demo focused on the Memory mechanism.
            for t in ctx_texts:
                enc2 = tokenizer(
                    [t],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(model_cfg.max_length),
                )
                with torch.no_grad():
                    _ = model(
                        input_ids=enc2["input_ids"].to(device),
                        attention_mask=enc2["attention_mask"].to(device),
                        user_ids=[effective_user_id],
                        update_memory=True,
                        gnn_context=None,
                        rag_context=None,
                    )

        out_all = _predict(rag_on=bool(rag_ctx is not None), gnn_on=bool(gnn_add is not None))
        out_no_rag = _predict(rag_on=False, gnn_on=bool(gnn_add is not None))
        out_no_gnn = _predict(rag_on=bool(rag_ctx is not None), gnn_on=False)

        # Demonstrate Memory effect by comparing prediction before vs after ingesting context.
        out_mem_before = None
        out_mem_after = None
        if ctx_texts and model.cfg.use_memory:
            # Snapshot 1: reset, predict without any prior context
            model.reset_streaming_state()
            out_mem_before = _predict(rag_on=False, gnn_on=False)
            # Snapshot 2: ingest context lines, then predict same text again
            _ingest_memory_context()
            out_mem_after = _predict(rag_on=False, gnn_on=False)

        risk_probs = _probs_from_logits(out_all["risk_logits"])
        emo_probs = _probs_from_logits(out_all["emotion_logits"])

        risk_classes = list(model_cfg.risk_classes)
        emo_classes = list(model_cfg.emotion_classes)

        risk_label, p_suicide = _label_risk(risk_probs, risk_classes, threshold_high)

        top_emo_idx = int(np.argmax(emo_probs)) if len(emo_probs) else -1
        emotion_top = emo_classes[top_emo_idx] if 0 <= top_emo_idx < len(emo_classes) else "unknown"
        top_emo_conf_pct = float(emo_probs[top_emo_idx] * 100.0) if 0 <= top_emo_idx < len(emo_probs) else 0.0

        with col2:
            st.subheader("Output")
            st.write(f"**Risk:** {risk_label} (p_suicide={p_suicide:.3f}, threshold={threshold_high:.2f})")
            st.write(f"**Top emotion:** {emotion_top} ({top_emo_conf_pct:.1f}% confidence)")
            st.caption("Interpretation: risk is a signal to prioritize review/support, not a diagnosis.")

            st.markdown("**Module effect (same input)**")
            rp_all = _probs_from_logits(out_all["risk_logits"])
            rp_no_rag = _probs_from_logits(out_no_rag["risk_logits"])
            rp_no_gnn = _probs_from_logits(out_no_gnn["risk_logits"])
            _, p_all = _label_risk(rp_all, risk_classes, threshold_high)
            _, p_nr = _label_risk(rp_no_rag, risk_classes, threshold_high)
            _, p_ng = _label_risk(rp_no_gnn, risk_classes, threshold_high)
            st.write(f"All modules: p_suicide={p_all:.3f}")
            st.write(f"No RAG: p_suicide={p_nr:.3f} (Δ={p_all - p_nr:+.3f})")
            st.write(f"No GNN: p_suicide={p_ng:.3f} (Δ={p_all - p_ng:+.3f})")

            if out_mem_before is not None and out_mem_after is not None:
                rp_b = _probs_from_logits(out_mem_before["risk_logits"])
                rp_a = _probs_from_logits(out_mem_after["risk_logits"])
                _, p_b = _label_risk(rp_b, risk_classes, threshold_high)
                _, p_a = _label_risk(rp_a, risk_classes, threshold_high)
                st.write(f"Memory demo (no RAG/GNN): before_context p_suicide={p_b:.3f}")
                st.write(f"Memory demo (no RAG/GNN): after_context p_suicide={p_a:.3f} (Δ={p_a - p_b:+.3f})")

            st.markdown("**Risk probabilities**")
            st.bar_chart({risk_classes[i]: float(risk_probs[i]) for i in range(len(risk_classes))})

            st.markdown("**Emotion confidence (%)**")
            st.bar_chart({emo_classes[i]: float(emo_probs[i] * 100.0) for i in range(len(emo_classes))})

            st.markdown("**Privacy-masked text used for inference**")
            st.code(clean_text)

            if show_debug:
                st.markdown("**Module debug**")
                if model.memory_store is not None and model.cfg.use_memory:
                    try:
                        long_state = model.memory_store.get_long(effective_user_id, device=device, dim=int(model.cfg.memory.user_state_dim))
                        st.write(f"Memory long_state L2 norm: {float(torch.linalg.vector_norm(long_state).item()):.3f}")
                    except Exception:
                        pass
                if rag_ctx is not None:
                    st.write(f"RAG ctx L2 norm: {float(torch.linalg.vector_norm(rag_ctx[0]).item()):.3f}")
                    if rag_texts:
                        with st.expander("Retrieved texts (top-k)"):
                            for t in rag_texts[:5]:
                                st.write(str(t)[:300])
                if gnn_add is not None:
                    st.write(f"GNN ctx L2 norm: {float(torch.linalg.vector_norm(gnn_add[0]).item()):.3f}")
                    st.write(f"GNN neighbors: {len(neigh_ids)}")
                    if neigh_ids:
                        with st.expander("Neighbor user_ids (sample)"):
                            for nid in neigh_ids[:16]:
                                st.write(str(nid))

    st.divider()
    # st.subheader("Next steps")
    # st.write(
    #     "- Train multitask (p_emotion>0) to make emotion head accurate.\n"
    #     "- Evaluate on test with scripts/evaluate.py and compare runs with scripts/aggregate_results.py."
    # )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
