from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.device import resolve_device
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


@st.cache_resource
def _load_model(config_path: str, ckpt_path: str, overrides: Tuple[str, ...]):
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

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    threshold_high = float(cfg.get("inference", {}).get("threshold_high", 0.55))

    return cfg, model_cfg, tokenizer, model, device, privacy_cfg, threshold_high


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default="configs/local_multitask.yaml")
    ap.add_argument("--ckpt", default="outputs/risk_only/best.pt")
    ap.add_argument("--overrides", nargs="*", default=None)
    args, _unknown = ap.parse_known_args()

    st.set_page_config(page_title="SRRTD Dashboard", layout="wide")

    st.title("Suicide Risk + Emotion Dashboard (Research Prototype)")
    st.caption("Not a diagnosis. Use for research/demo only.")

    with st.sidebar:
        st.subheader("Model")
        config_path = st.text_input("Config path", value=str(args.config))
        ckpt_path = st.text_input("Checkpoint path", value=str(args.ckpt))
        overrides_txt = st.text_area("Overrides (space-separated key=value)", value=" ".join(args.overrides or []))
        overrides = tuple([s for s in overrides_txt.split() if s.strip()])

        st.subheader("Session")
        user_id = st.text_input("user_id", value="demo_user")
        update_memory = st.checkbox("Update memory (streaming context)", value=True)

    try:
        cfg, model_cfg, tokenizer, model, device, privacy_cfg, threshold_high = _load_model(config_path, ckpt_path, overrides)
    except Exception as e:
        st.error(f"Failed to load model/checkpoint: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        text = st.text_area("Text", height=180, value="I feel hopeless and I don't know what to do.")
        run = st.button("Predict")

    if run:
        clean_text = privacy_preprocess(text, privacy_cfg)

        enc = tokenizer(
            [clean_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=[user_id],
                update_memory=bool(update_memory),
                gnn_context=None,
                rag_context=None,
            )

        risk_probs = torch.softmax(out["risk_logits"], dim=-1)[0].detach().cpu().numpy()
        emo_probs = torch.softmax(out["emotion_logits"], dim=-1)[0].detach().cpu().numpy()

        risk_classes = list(model_cfg.risk_classes)
        emo_classes = list(model_cfg.emotion_classes)

        suicide_idx = 1 if len(risk_classes) > 1 else 0
        p_suicide = float(risk_probs[suicide_idx])
        risk_label = "HIGH" if p_suicide >= float(threshold_high) else "LOW"

        top_emo_idx = int(np.argmax(emo_probs)) if len(emo_probs) else -1
        emotion_top = emo_classes[top_emo_idx] if 0 <= top_emo_idx < len(emo_classes) else "unknown"

        with col2:
            st.subheader("Output")
            st.write(f"**Risk:** {risk_label} (p_suicide={p_suicide:.3f}, threshold={threshold_high:.2f})")
            st.write(f"**Top emotion:** {emotion_top}")
            st.caption("Interpretation: risk is a signal to prioritize review/support, not a diagnosis.")

            st.markdown("**Risk probabilities**")
            st.bar_chart({risk_classes[i]: float(risk_probs[i]) for i in range(len(risk_classes))})

            st.markdown("**Emotion probabilities**")
            st.bar_chart({emo_classes[i]: float(emo_probs[i]) for i in range(len(emo_classes))})

            st.markdown("**Privacy-masked text used for inference**")
            st.code(clean_text)

    st.divider()
    st.subheader("Next steps")
    st.write(
        "- Train multitask (p_emotion>0) to make emotion head accurate.\n"
        "- Evaluate on test with scripts/evaluate.py and compare runs with scripts/aggregate_results.py."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
