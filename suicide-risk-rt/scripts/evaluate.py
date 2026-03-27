from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    from transformers.models.auto.tokenization_auto import AutoTokenizer

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle, load_multitask_bundles
from srrtd.data.torch_dataset import StreamingPostDataset, collate_tokenized
from srrtd.eval.metrics import compute_emotion_metrics, compute_metrics, compute_risk_metrics
from srrtd.models.community import CommunityGraph, CommunityGnnContext
from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.rag.retriever import RagConfig, RagRetriever
from srrtd.utils.config import apply_overrides, load_yaml, resolve_paths
from srrtd.utils.device import resolve_device
from srrtd.utils.seed import set_global_seed


def _plot_confusion(cm: list[list[int]], classes: list[str], title: str, path: Path) -> None:
    arr = np.asarray(cm)
    plt.figure(figsize=(7, 6))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--ckpt", default="outputs/run/best.pt")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(cfg, root)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    mode = str(cfg.get("data", {}).get("mode", "single")).lower()
    try:
        if mode == "multitask":
            risk_bundle, emo_bundle = load_multitask_bundles(cfg, root)
            bundle = None
        else:
            bundle = load_dataset_bundle(cfg, root)
            risk_bundle, emo_bundle = None, None
    except DatasetConfirmationRequired as e:
        raise SystemExit(str(e))

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    ckpt_path = (root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Legacy checkpoints (older runs) may contain non-tensor python objects.
        # This fallback is safe when the checkpoint is locally produced by this repo.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_cfg = model_config_from_yaml(cfg)
    # align heads to checkpoint label sizes
    model_cfg = model_cfg.__class__(**{**model_cfg.__dict__, "risk_classes": ckpt["risk_classes"], "emotion_classes": ckpt["emotion_classes"]})

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt.get("tokenizer_name", model_cfg.encoder_name)))
    model = SrrtdMultiTaskModel(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    rag = None
    if model_cfg.use_rag:
        rag_cfg_d = dict(cfg.get("model", {}).get("rag", {}) or {})
        rag_cfg = RagConfig(
            embed_model=str(rag_cfg_d.get("embed_model")),
            embed_dim=int(rag_cfg_d.get("embed_dim", model_cfg.rag_embed_dim)),
            top_k=int(rag_cfg_d.get("top_k", 5)),
            chroma_dir=str(rag_cfg_d.get("chroma_dir")),
            collection=str(rag_cfg_d.get("collection")),
        )
        rag = RagRetriever(rag_cfg, root)

    gnn_ctx = None
    graph = None
    if model_cfg.use_gnn:
        # Prefer the exact graph used during training if saved next to the checkpoint.
        graph_json = ckpt_path.parent / "community_graph.json"
        if graph_json.exists():
            try:
                with open(graph_json, "r", encoding="utf-8") as f:
                    g = json.load(f)
                if isinstance(g, dict) and isinstance(g.get("adj"), dict):
                    graph = CommunityGraph()
                    graph.adj = {str(k): [str(x) for x in (v or [])] for k, v in (g.get("adj") or {}).items()}
            except Exception:
                graph = None

        recs = []
        if mode == "multitask":
            for split in (risk_bundle.train, risk_bundle.val, risk_bundle.test):
                for r in split.records:
                    recs.append({"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp})
            for split in (emo_bundle.train, emo_bundle.val, emo_bundle.test):
                for r in split.records:
                    recs.append({"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp})
        else:
            for split in (bundle.train, bundle.val, bundle.test):
                for r in split.records:
                    recs.append({"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp})
        if graph is None:
            graph = CommunityGraph.build_from_records(
                records=recs,
                seed=seed,
                max_neighbors=int(cfg.get("model", {}).get("gnn", {}).get("max_neighbors", 16)),
            )
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
        else:
            # Avoid using random GNN weights during evaluation.
            print("WARN: use_gnn=true but checkpoint missing gnn_state_dict; disabling GNN for evaluation.")
            gnn_ctx = None
            graph = None
        if gnn_ctx is not None:
            gnn_ctx.eval()

    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cfg = dict(cfg.get("train", {}) or {})
    bs = int(test_cfg.get("batch_size", 8))

    collate = collate_tokenized(tokenizer, max_length=model_cfg.max_length)

    def _run_split(ds: StreamingPostDataset, desc: str):
        dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
        risk_logits_l, emo_logits_l, risk_y_l, emo_y_l = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dl, desc=desc):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                risk_y = batch["risk"].to(device)
                emo_y = batch["emotion"].to(device)
                user_ids = batch["user_ids"]
                texts = batch["texts"]

                rag_ctx = rag.retrieve_context_vec(texts, device=device) if rag is not None else None

                gnn_add = None
                if gnn_ctx is not None and graph is not None and model.memory_store is not None:
                    gnn_add = gnn_ctx(
                        user_ids=user_ids,
                        device=device,
                        graph=graph,
                        user_state_lookup=lambda uid, dev, dim: model.memory_store.get_long(uid, dev, dim),
                    )

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    user_ids=user_ids,
                    update_memory=True,
                    gnn_context=gnn_add,
                    rag_context=rag_ctx,
                )

                risk_logits_l.append(out["risk_logits"].detach().cpu().numpy())
                emo_logits_l.append(out["emotion_logits"].detach().cpu().numpy())
                risk_y_l.append(risk_y.detach().cpu().numpy())
                emo_y_l.append(emo_y.detach().cpu().numpy())
        return (
            np.concatenate(risk_logits_l, axis=0),
            np.concatenate(emo_logits_l, axis=0),
            np.concatenate(risk_y_l, axis=0),
            np.concatenate(emo_y_l, axis=0),
        )

    model.reset_streaming_state()

    if mode == "multitask":
        # Risk evaluation
        risk_logits, emo_logits, risk_y, emo_y = _run_split(StreamingPostDataset(risk_bundle.test), "test risk")
        rb, rdetails = compute_risk_metrics(risk_logits, risk_y, ckpt["risk_classes"])
        _plot_confusion(rdetails["risk_confusion"], ckpt["risk_classes"], "Risk Confusion", out_dir / "risk_confusion.png")

        # Emotion evaluation
        risk_logits2, emo_logits2, risk_y2, emo_y2 = _run_split(StreamingPostDataset(emo_bundle.test), "test emotion")
        eb, edetails = compute_emotion_metrics(emo_logits2, emo_y2, ckpt["emotion_classes"])
        _plot_confusion(
            edetails["emotion_confusion"],
            ckpt["emotion_classes"],
            "Emotion Confusion",
            out_dir / "emotion_confusion.png",
        )

        metrics = {
            "risk_f1_macro": rb.risk_f1_macro,
            "risk_f1_weighted": rb.risk_f1_weighted,
            "risk_recall_high": rb.risk_recall_high,
            "risk_roc_auc_ovr": rb.risk_roc_auc_ovr,
            "emotion_f1_macro": eb.emotion_f1_macro,
            "emotion_acc": eb.emotion_acc,
        }
        details = {**rdetails, **edetails}
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "details": details}, f, indent=2)

        print("OK:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        ds_test = StreamingPostDataset(bundle.test)
        risk_logits, emo_logits, risk_y, emo_y = _run_split(ds_test, "test")
        mb, details = compute_metrics(risk_logits, risk_y, emo_logits, emo_y, ckpt["risk_classes"], ckpt["emotion_classes"])
        metrics = {
            "risk_f1_macro": mb.risk_f1_macro,
            "risk_f1_weighted": mb.risk_f1_weighted,
            "risk_recall_high": mb.risk_recall_high,
            "risk_roc_auc_ovr": mb.risk_roc_auc_ovr,
            "emotion_f1_macro": mb.emotion_f1_macro,
            "emotion_acc": mb.emotion_acc,
        }

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "details": details}, f, indent=2)

        _plot_confusion(details["risk_confusion"], ckpt["risk_classes"], "Risk Confusion", out_dir / "risk_confusion.png")
        _plot_confusion(details["emotion_confusion"], ckpt["emotion_classes"], "Emotion Confusion", out_dir / "emotion_confusion.png")

        print("OK:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
