from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
from pathlib import Path
from itertools import cycle
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:  # pragma: no cover
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

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
    ap.add_argument(
        "--init-ckpt",
        default=None,
        help="Optional checkpoint path to initialize weights from (state_dict is loaded with strict=False).",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(cfg, root)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    mode = str(cfg.get("data", {}).get("mode", "single")).lower()
    try:
        print(f"DATA: mode={mode} loading bundles...")
        if mode == "multitask":
            risk_bundle, emo_bundle = load_multitask_bundles(cfg, root)
            bundle = None
        else:
            bundle = load_dataset_bundle(cfg, root)
            risk_bundle, emo_bundle = None, None
    except DatasetConfirmationRequired as e:
        raise SystemExit(str(e))

    if mode == "multitask":
        print(
            "DATA: loaded multitask bundles "
            f"risk_train={len(risk_bundle.train.records)} emotion_train={len(emo_bundle.train.records)}"
        )
    else:
        print(f"DATA: loaded single bundle train={len(bundle.train.records)}")

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    model_cfg = model_config_from_yaml(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.encoder_name)

    model = SrrtdMultiTaskModel(model_cfg).to(device)
    model.train()

    if args.init_ckpt:
        print(f"INIT: loading init checkpoint '{args.init_ckpt}'...")
        init_path = (root / str(args.init_ckpt)).resolve() if not Path(str(args.init_ckpt)).is_absolute() else Path(str(args.init_ckpt))
        try:
            init_ckpt = torch.load(init_path, map_location="cpu", weights_only=True)
        except Exception:
            init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)

        if isinstance(init_ckpt, dict) and "state_dict" in init_ckpt:
            # Helpful sanity checks: warn if label spaces disagree.
            ckpt_risk = list(init_ckpt.get("risk_classes", []) or [])
            ckpt_emo = list(init_ckpt.get("emotion_classes", []) or [])
            if ckpt_risk and ckpt_risk != list(model_cfg.risk_classes):
                print(
                    "WARN: init checkpoint risk_classes differ from current config. "
                    f"ckpt={ckpt_risk} cfg={list(model_cfg.risk_classes)}"
                )
            if ckpt_emo and ckpt_emo != list(model_cfg.emotion_classes):
                print(
                    "WARN: init checkpoint emotion_classes differ from current config. "
                    f"ckpt={ckpt_emo} cfg={list(model_cfg.emotion_classes)}"
                )

            # Robust partial load: skip any tensors whose shapes don't match the current model.
            # This is critical when emotion/risk label spaces differ across runs.
            cur_sd = model.state_dict()
            raw_sd = init_ckpt["state_dict"]
            filtered_sd = {}
            skipped = []
            for k, v in raw_sd.items():
                if k not in cur_sd:
                    continue
                try:
                    if hasattr(v, "shape") and hasattr(cur_sd[k], "shape") and tuple(v.shape) != tuple(cur_sd[k].shape):
                        skipped.append(k)
                        continue
                except Exception:
                    # If shape check fails for any reason, skip conservatively.
                    skipped.append(k)
                    continue
                filtered_sd[k] = v

            load_res = model.load_state_dict(filtered_sd, strict=False)
            missing = getattr(load_res, "missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", [])
            print(
                "INIT_CKPT loaded (filtered by shape): "
                f"loaded={len(filtered_sd)} skipped_shape_mismatch={len(skipped)} "
                f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
            )
        else:
            raise SystemExit(f"Invalid init checkpoint format (missing 'state_dict'): {init_path}")

    # Optional RAG
    rag = None
    if model_cfg.use_rag:
        print("RAG: initializing retriever (this may take time on first run)...")
        rag_cfg_d = dict(cfg.get("model", {}).get("rag", {}) or {})
        rag_cfg = RagConfig(
            embed_model=str(rag_cfg_d.get("embed_model")),
            embed_dim=int(rag_cfg_d.get("embed_dim", model_cfg.rag_embed_dim)),
            top_k=int(rag_cfg_d.get("top_k", 5)),
            chroma_dir=str(rag_cfg_d.get("chroma_dir")),
            collection=str(rag_cfg_d.get("collection")),
        )
        rag = RagRetriever(rag_cfg, root)
        try:
            if hasattr(rag.collection, "count"):
                print(f"RAG: collection='{rag_cfg.collection}' count={int(rag.collection.count())}")
        except Exception:
            print(f"RAG: collection='{rag_cfg.collection}' ready")

    # Optional GNN
    gnn_ctx = None
    graph = None
    if model_cfg.use_gnn:
        print("GNN: building community graph...")
        if mode == "multitask":
            recs = _as_records(risk_bundle) + _as_records(emo_bundle)
        else:
            recs = _as_records(bundle)
        graph = CommunityGraph.build_from_records(
            records=recs,
            seed=seed,
            max_neighbors=int(cfg.get("model", {}).get("gnn", {}).get("max_neighbors", 16)),
        )
        try:
            print(f"GNN: graph users={len(graph.adj)}")
        except Exception:
            pass
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
    max_steps = int(train_cfg.get("max_steps", -1))

    loss_w = dict(train_cfg.get("loss_weights", {}) or {})
    w_risk = float(loss_w.get("risk", 1.0))
    w_emo = float(loss_w.get("emotion", 0.5))

    collate = collate_tokenized(tokenizer, max_length=model_cfg.max_length)
    if mode == "multitask":
        ds_train_risk = StreamingPostDataset(risk_bundle.train)
        ds_val_risk = StreamingPostDataset(risk_bundle.val)
        ds_train_emo = StreamingPostDataset(emo_bundle.train)
        ds_val_emo = StreamingPostDataset(emo_bundle.val)

        dl_train_risk = DataLoader(ds_train_risk, batch_size=bs, shuffle=False, collate_fn=collate)
        dl_val_risk = DataLoader(ds_val_risk, batch_size=bs, shuffle=False, collate_fn=collate)
        dl_train_emo = DataLoader(ds_train_emo, batch_size=bs, shuffle=False, collate_fn=collate)
        dl_val_emo = DataLoader(ds_val_emo, batch_size=bs, shuffle=False, collate_fn=collate)
    else:
        ds_train = StreamingPostDataset(bundle.train)
        ds_val = StreamingPostDataset(bundle.val)
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=False, collate_fn=collate)
        dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, collate_fn=collate)

    params = list(model.parameters())
    if gnn_ctx is not None:
        params += list(gnn_ctx.parameters())

    opt = AdamW(params, lr=lr, weight_decay=wd)

    if mode == "multitask":
        steps_per_epoch_raw = int(cfg.get("train", {}).get("multitask", {}).get("steps_per_epoch", -1))
        if steps_per_epoch_raw > 0:
            steps_per_epoch = steps_per_epoch_raw
        else:
            steps_per_epoch = max(1, len(dl_train_risk))
        steps_per_epoch = max(1, int(np.ceil(steps_per_epoch / max(1, grad_accum))))
    else:
        steps_per_epoch = max(1, int(np.ceil(len(dl_train) / max(1, grad_accum))))
    total_steps = steps_per_epoch * max(1, epochs)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = min(warmup_steps, total_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    ce = torch.nn.CrossEntropyLoss()

    run_dir = paths.outputs / str(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist community graph used for this run (so API/dashboard can reproduce GNN behavior
    # without rebuilding from the full dataset).
    if graph is not None:
        try:
            with open(run_dir / "community_graph.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 2,
                        "meta": {
                            "seed": int(seed),
                            "max_neighbors": int(cfg.get("model", {}).get("gnn", {}).get("max_neighbors", 16)),
                        },
                        "adj": graph.adj,
                    },
                    f,
                )
        except Exception:
            # Non-fatal: training can proceed even if graph serialization fails.
            pass

    best_val = -1.0
    best_epoch = -1
    best_metrics: dict | None = None
    global_step = 0
    reached_max_steps = False

    rng = random.Random(seed)
    p_emo = float(cfg.get("train", {}).get("multitask", {}).get("p_emotion", 0.25))
    p_emo = max(0.0, min(1.0, p_emo))

    def _infinite(loader):
        while True:
            for b in loader:
                yield b

    if mode == "multitask":
        risk_iter = _infinite(dl_train_risk)
        emo_iter = _infinite(dl_train_emo)

    for epoch in range(1, epochs + 1):
        print(f"EPOCH {epoch}/{epochs}")
        model.train()
        if gnn_ctx is not None:
            gnn_ctx.train()
        model.reset_streaming_state()

        if mode == "multitask":
            pbar = tqdm(range(1, steps_per_epoch * max(1, grad_accum) + 1), desc=f"train epoch {epoch}")
        else:
            pbar = tqdm(dl_train, desc=f"train epoch {epoch}")
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            if mode == "multitask":
                train_task = "emotion" if rng.random() < p_emo else "risk"
                batch = next(emo_iter) if train_task == "emotion" else next(risk_iter)
            else:
                train_task = "both"

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

            if train_task == "risk":
                loss_r = ce(out["risk_logits"], risk_y)
                loss = w_risk * loss_r
            elif train_task == "emotion":
                loss_e = ce(out["emotion_logits"], emo_y)
                loss = w_emo * loss_e
            else:
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

                if max_steps > 0 and global_step >= max_steps:
                    reached_max_steps = True
                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix({"loss": float(loss.item() * max(1, grad_accum)), "task": train_task, "stop": "max_steps"})
                    break

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": float(loss.item() * max(1, grad_accum)), "task": train_task})

        if reached_max_steps:
            print(f"Reached max_steps={max_steps} at epoch={epoch} (global_step={global_step}).")

        # Validation
        model.eval()
        if gnn_ctx is not None:
            gnn_ctx.eval()

        if mode == "multitask":
            # Validate risk on risk val
            val_risk_logits, val_risk_y = [], []
            with torch.no_grad():
                for batch in tqdm(dl_val_risk, desc=f"val risk epoch {epoch}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    risk_y = batch["risk"].to(device)
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
                        update_memory=False,
                        gnn_context=gnn_add,
                        rag_context=rag_ctx,
                    )
                    val_risk_logits.append(out["risk_logits"].detach().cpu().numpy())
                    val_risk_y.append(risk_y.detach().cpu().numpy())

            risk_logits = np.concatenate(val_risk_logits, axis=0)
            risk_y = np.concatenate(val_risk_y, axis=0)
            rb, rdetails = compute_risk_metrics(risk_logits, risk_y, risk_bundle.risk_classes)

            # Validate emotion on emotion val
            val_emo_logits, val_emo_y = [], []
            with torch.no_grad():
                for batch in tqdm(dl_val_emo, desc=f"val emotion epoch {epoch}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
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
                        update_memory=False,
                        gnn_context=gnn_add,
                        rag_context=rag_ctx,
                    )
                    val_emo_logits.append(out["emotion_logits"].detach().cpu().numpy())
                    val_emo_y.append(emo_y.detach().cpu().numpy())

            emo_logits = np.concatenate(val_emo_logits, axis=0)
            emo_y = np.concatenate(val_emo_y, axis=0)
            eb, edetails = compute_emotion_metrics(emo_logits, emo_y, emo_bundle.emotion_classes)

            metrics = {
                "epoch": epoch,
                "risk_f1_macro": rb.risk_f1_macro,
                "risk_f1_weighted": rb.risk_f1_weighted,
                "risk_recall_high": rb.risk_recall_high,
                "risk_roc_auc_ovr": rb.risk_roc_auc_ovr,
                "emotion_f1_macro": eb.emotion_f1_macro,
                "emotion_acc": eb.emotion_acc,
            }
            details = {**rdetails, **edetails}
            with open(run_dir / f"val_metrics_epoch_{epoch}.json", "w", encoding="utf-8") as f:
                json.dump({"metrics": metrics, "details": details}, f, indent=2)

            score = float(metrics["risk_f1_macro"] + metrics["emotion_f1_macro"]) / 2.0
        else:
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
                "risk_classes": list((risk_bundle or bundle).risk_classes),
                "emotion_classes": list((emo_bundle or bundle).emotion_classes),
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
                "gnn_state_dict": (gnn_ctx.state_dict() if gnn_ctx is not None else None),
                "risk_classes": (risk_bundle or bundle).risk_classes,
                "emotion_classes": (emo_bundle or bundle).emotion_classes,
                "tokenizer_name": model_cfg.encoder_name,
            }
            torch.save(ckpt, run_dir / "best.pt")

        if mode == "multitask":
            print(
                f"VAL epoch={epoch} risk_f1_macro={metrics['risk_f1_macro']:.4f} "
                f"emotion_f1_macro={metrics['emotion_f1_macro']:.4f}"
            )
        else:
            print(f"VAL epoch={epoch} risk_f1_macro={mb.risk_f1_macro:.4f} recall_high={mb.risk_recall_high:.4f}")

        if reached_max_steps:
            break

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
            "mode": mode,
            "source": str(cfg.get("data", {}).get("source", "")),
        },
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
