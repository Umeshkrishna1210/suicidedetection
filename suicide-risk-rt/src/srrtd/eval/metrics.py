from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    risk_f1_macro: float
    risk_f1_weighted: float
    risk_recall_high: float
    risk_roc_auc_ovr: float | None
    risk_acc: float
    emotion_f1_macro: float
    emotion_acc: float


@dataclass(frozen=True)
class RiskMetricBundle:
    risk_f1_macro: float
    risk_f1_weighted: float
    risk_recall_high: float
    risk_roc_auc_ovr: float | None
    risk_acc: float


@dataclass(frozen=True)
class EmotionMetricBundle:
    emotion_f1_macro: float
    emotion_acc: float


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(
    risk_logits: np.ndarray,
    risk_y: np.ndarray,
    emotion_logits: np.ndarray,
    emotion_y: np.ndarray,
    risk_classes: list[str],
    emotion_classes: list[str],
) -> tuple[MetricBundle, dict[str, Any]]:
    risk_labels = list(range(len(risk_classes)))
    emotion_labels = list(range(len(emotion_classes)))

    risk_pred = risk_logits.argmax(axis=-1)
    emo_pred = emotion_logits.argmax(axis=-1)

    risk_f1_macro = float(f1_score(risk_y, risk_pred, average="macro", labels=risk_labels, zero_division=0))
    risk_f1_weighted = float(f1_score(risk_y, risk_pred, average="weighted", labels=risk_labels, zero_division=0))
    high_idx = len(risk_classes) - 1
    risk_recall_high = float(recall_score((risk_y == high_idx).astype(int), (risk_pred == high_idx).astype(int)))
    risk_acc = float(accuracy_score(risk_y, risk_pred))

    # ROC AUC OVR if possible
    risk_auc: float | None
    try:
        probs = _softmax(risk_logits)
        risk_auc = float(roc_auc_score(risk_y, probs, multi_class="ovr"))
    except Exception:
        risk_auc = None

    emo_f1_macro = float(f1_score(emotion_y, emo_pred, average="macro", labels=emotion_labels, zero_division=0))
    emo_acc = float(accuracy_score(emotion_y, emo_pred))

    details = {
        "risk_report": classification_report(
            risk_y,
            risk_pred,
            labels=risk_labels,
            target_names=risk_classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "emotion_report": classification_report(
            emotion_y,
            emo_pred,
            labels=emotion_labels,
            target_names=emotion_classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "risk_confusion": confusion_matrix(risk_y, risk_pred, labels=risk_labels).tolist(),
        "emotion_confusion": confusion_matrix(emotion_y, emo_pred, labels=emotion_labels).tolist(),
    }

    return (
        MetricBundle(
            risk_f1_macro=risk_f1_macro,
            risk_f1_weighted=risk_f1_weighted,
            risk_recall_high=risk_recall_high,
            risk_roc_auc_ovr=risk_auc,
            risk_acc=risk_acc,
            emotion_f1_macro=emo_f1_macro,
            emotion_acc=emo_acc,
        ),
        details,
    )


def compute_risk_metrics(
    risk_logits: np.ndarray,
    risk_y: np.ndarray,
    risk_classes: list[str],
) -> tuple[RiskMetricBundle, dict[str, Any]]:
    risk_labels = list(range(len(risk_classes)))
    risk_pred = risk_logits.argmax(axis=-1)

    risk_f1_macro = float(f1_score(risk_y, risk_pred, average="macro", labels=risk_labels, zero_division=0))
    risk_f1_weighted = float(f1_score(risk_y, risk_pred, average="weighted", labels=risk_labels, zero_division=0))
    high_idx = len(risk_classes) - 1
    risk_recall_high = float(recall_score((risk_y == high_idx).astype(int), (risk_pred == high_idx).astype(int)))
    risk_acc = float(accuracy_score(risk_y, risk_pred))

    risk_auc: float | None
    try:
        probs = _softmax(risk_logits)
        risk_auc = float(roc_auc_score(risk_y, probs, multi_class="ovr"))
    except Exception:
        risk_auc = None

    details = {
        "risk_report": classification_report(
            risk_y,
            risk_pred,
            labels=risk_labels,
            target_names=risk_classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "risk_confusion": confusion_matrix(risk_y, risk_pred, labels=risk_labels).tolist(),
    }

    return (
        RiskMetricBundle(
            risk_f1_macro=risk_f1_macro,
            risk_f1_weighted=risk_f1_weighted,
            risk_recall_high=risk_recall_high,
            risk_roc_auc_ovr=risk_auc,
            risk_acc=risk_acc,
        ),
        details,
    )


def compute_emotion_metrics(
    emotion_logits: np.ndarray,
    emotion_y: np.ndarray,
    emotion_classes: list[str],
) -> tuple[EmotionMetricBundle, dict[str, Any]]:
    emotion_labels = list(range(len(emotion_classes)))
    emo_pred = emotion_logits.argmax(axis=-1)
    emo_f1_macro = float(f1_score(emotion_y, emo_pred, average="macro", labels=emotion_labels, zero_division=0))
    emo_acc = float(accuracy_score(emotion_y, emo_pred))

    details = {
        "emotion_report": classification_report(
            emotion_y,
            emo_pred,
            labels=emotion_labels,
            target_names=emotion_classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "emotion_confusion": confusion_matrix(emotion_y, emo_pred, labels=emotion_labels).tolist(),
    }

    return EmotionMetricBundle(emotion_f1_macro=emo_f1_macro, emotion_acc=emo_acc), details
