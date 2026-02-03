from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle
from srrtd.data.torch_dataset import StreamingPostDataset, collate_tokenized
from srrtd.eval.metrics import compute_metrics
from srrtd.models.community import CommunityGraph, CommunityGnnContext
from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.rag.retriever import RagConfig, RagRetriever
from srrtd.utils.config import apply_overrides, load_yaml, resolve_paths
from srrtd.utils.device import resolve_device
from srrtd.utils.seed import set_global_seed


def _as_records(bundle) -> list[dict]:
    rows = []
    for split in (bundle.train, bundle.val, bundle.test):
        for r in split.records:
            rows.append({"user_id": r.user_id, "lang": r.lang, "timestamp": r.timestamp})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--run-name", default="run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(cfg, root)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    try:
        bundle = load_dataset_bundle(cfg, root)
    except DatasetConfirmationRequired as e:
        raise SystemExit(str(e))

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    model_cfg = model_config_from_yaml(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.encoder_name)

    model = SrrtdMultiTaskModel(model_cfg).to(device)
    model.train()

    # Optional RAG
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

    # Optional GNN
    gnn_ctx = None
    graph = None
    if model_cfg.use_gnn:
        recs = _as_records(bundle)
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
        gnn_ctx.train()

    train_cfg = dict(cfg.get("train", {}) or {})
    bs = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("lr", 2e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))
    epochs = int(train_cfg.get("epochs", 1))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))

    loss_w = dict(train_cfg.get("loss_weights", {}) or {})
    w_risk = float(loss_w.get("risk", 1.0))
    w_emo = float(loss_w.get("emotion", 0.5))

    ds_train = StreamingPostDataset(bundle.train)
    ds_val = StreamingPostDataset(bundle.val)

    collate = collate_tokenized(tokenizer, max_length=model_cfg.max_length)
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=False, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, collate_fn=collate)

    params = list(model.parameters())
    if gnn_ctx is not None:
        params += list(gnn_ctx.parameters())

    opt = AdamW(params, lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, int(np.ceil(len(dl_train) / max(1, grad_accum))))
    total_steps = steps_per_epoch * max(1, epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    ce = torch.nn.CrossEntropyLoss()

    run_dir = paths.outputs / str(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_epoch = -1
    best_metrics: dict | None = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if gnn_ctx is not None:
            gnn_ctx.train()
        model.reset_streaming_state()

        pbar = tqdm(dl_train, desc=f"train epoch {epoch}")
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            risk_y = batch["risk"].to(device)
            emo_y = batch["emotion"].to(device)
            user_ids = batch["user_ids"]
            texts = batch["texts"]

            rag_ctx = None
            if rag is not None:
                rag_ctx = rag.retrieve_context_vec(texts, device=device)

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

            loss_r = ce(out["risk_logits"], risk_y)
            loss_e = ce(out["emotion_logits"], emo_y)
            loss = w_risk * loss_r + w_emo * loss_e
            loss = loss / max(1, grad_accum)
            loss.backward()

            if step % max(1, grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({"loss": float(loss.item() * max(1, grad_accum))})

        # Validation
        model.eval()
        if gnn_ctx is not None:
            gnn_ctx.eval()

        val_risk_logits = []
        val_emo_logits = []
        val_risk_y = []
        val_emo_y = []

        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"val epoch {epoch}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                risk_y = batch["risk"].to(device)
                emo_y = batch["emotion"].to(device)
                user_ids = batch["user_ids"]
                texts = batch["texts"]

                rag_ctx = None
                if rag is not None:
                    rag_ctx = rag.retrieve_context_vec(texts, device=device)

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
                    update_memory=False,
                    gnn_context=gnn_add,
                    rag_context=rag_ctx,
                )

                val_risk_logits.append(out["risk_logits"].detach().cpu().numpy())
                val_emo_logits.append(out["emotion_logits"].detach().cpu().numpy())
                val_risk_y.append(risk_y.detach().cpu().numpy())
                val_emo_y.append(emo_y.detach().cpu().numpy())

        risk_logits = np.concatenate(val_risk_logits, axis=0)
        emo_logits = np.concatenate(val_emo_logits, axis=0)
        risk_y = np.concatenate(val_risk_y, axis=0)
        emo_y = np.concatenate(val_emo_y, axis=0)

        mb, details = compute_metrics(risk_logits, risk_y, emo_logits, emo_y, bundle.risk_classes, bundle.emotion_classes)
        metrics = {
            "epoch": epoch,
            "risk_f1_macro": mb.risk_f1_macro,
            "risk_f1_weighted": mb.risk_f1_weighted,
            "risk_recall_high": mb.risk_recall_high,
            "risk_roc_auc_ovr": mb.risk_roc_auc_ovr,
            "emotion_f1_macro": mb.emotion_f1_macro,
            "emotion_acc": mb.emotion_acc,
        }

        with open(run_dir / f"val_metrics_epoch_{epoch}.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "details": details}, f, indent=2)

        score = float(mb.risk_f1_macro)
        if score > best_val:
            best_val = score
            best_epoch = int(epoch)
            best_metrics = dict(metrics)
            safe_model_cfg = {
                "encoder_name": model_cfg.encoder_name,
                "max_length": int(model_cfg.max_length),
                "dropout": float(model_cfg.dropout),
                "risk_classes": list(bundle.risk_classes),
                "emotion_classes": list(bundle.emotion_classes),
                "use_memory": bool(model_cfg.use_memory),
                "memory": {
                    "short_window": int(model_cfg.memory.short_window),
                    "long_ema_decay": float(model_cfg.memory.long_ema_decay),
                    "user_state_dim": int(model_cfg.memory.user_state_dim),
                },
                "use_rag": bool(model_cfg.use_rag),
                "rag_embed_dim": int(model_cfg.rag_embed_dim),
                "use_gnn": bool(model_cfg.use_gnn),
                "gnn": {
                    "hidden_dim": int(model_cfg.gnn.hidden_dim),
                    "num_layers": int(model_cfg.gnn.num_layers),
                    "dropout": float(model_cfg.gnn.dropout),
                },
            }
            ckpt = {
                "version": 1,
                "model_cfg": safe_model_cfg,
                "state_dict": model.state_dict(),
                "risk_classes": bundle.risk_classes,
                "emotion_classes": bundle.emotion_classes,
                "tokenizer_name": model_cfg.encoder_name,
            }
            torch.save(ckpt, run_dir / "best.pt")

        print(f"VAL epoch={epoch} risk_f1_macro={mb.risk_f1_macro:.4f} recall_high={mb.risk_recall_high:.4f}")

    print(f"OK: saved {run_dir / 'best.pt'}")

    # Write a stable summary file for aggregation.
    if best_metrics is None:
        best_metrics = {"epoch": best_epoch}
    summary = {
        "run_name": str(args.run_name),
        "best_epoch": int(best_epoch),
        "metrics": best_metrics,
        "model": {
            "encoder_name": model_cfg.encoder_name,
            "use_memory": bool(model_cfg.use_memory),
            "use_rag": bool(model_cfg.use_rag),
            "use_gnn": bool(model_cfg.use_gnn),
        },
        "data": {
            "source": str(cfg.get("data", {}).get("source", "")),
        },
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
