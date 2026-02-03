from __future__ import annotations

import random
from typing import Any

from srrtd.data.schema import DatasetBundle, DatasetSplit, PostRecord


def build_toy_dataset(
    seed: int,
    risk_classes: list[str],
    emotion_classes: list[str],
    n_users: int = 50,
    posts_per_user: int = 12,
) -> DatasetBundle:
    rng = random.Random(seed)

    # Minimal multilingual, social-media-like snippets (synthetic)
    templates: list[tuple[str, str, int, int]] = [
        ("en", "Feeling overwhelmed lately. Nothing seems to work.", 1, emotion_classes.index("sadness")),
        ("en", "I can't do this anymore. I'm tired of everything.", 2, emotion_classes.index("sadness")),
        ("en", "I hate myself and I feel trapped.", 2, emotion_classes.index("fear")),
        ("en", "Had a good day. Grateful for small wins.", 0, emotion_classes.index("joy")),
        ("en", "I'm angry and frustrated at everyone.", 1, emotion_classes.index("anger")),
        ("es", "Me siento solo y sin esperanza.", 2, emotion_classes.index("sadness")),
        ("es", "Hoy fue un buen día, gracias a todos.", 0, emotion_classes.index("joy")),
        ("hi", "आज मन बहुत भारी है, समझ नहीं आ रहा क्या करूँ।", 1, emotion_classes.index("sadness")),
        ("hi", "सब ठीक है, मैं संभाल लूँगा।", 0, emotion_classes.index("neutral")),
        ("fr", "Je me sens épuisé et inutile.", 1, emotion_classes.index("sadness")),
        ("fr", "Rien ne va, je n'en peux plus.", 2, emotion_classes.index("fear")),
        ("ar", "أشعر بالحزن والوحدة.", 1, emotion_classes.index("sadness")),
        ("ar", "لا أستطيع الاستمرار هكذا.", 2, emotion_classes.index("fear")),
        ("en", "I feel okay. Just a normal day.", 0, emotion_classes.index("neutral")),
    ]

    records: list[PostRecord] = []
    base_ts = 1_700_000_000

    for u in range(n_users):
        user_id = f"user_{u:04d}"
        # Each user has a latent risk tendency
        user_risk_bias = rng.choices([0, 1, 2], weights=[0.65, 0.25, 0.10])[0]
        for p in range(posts_per_user):
            lang, text, risk_hint, emo = rng.choice(templates)
            # Mix per-user risk tendency with per-post hint
            risk = int(round((user_risk_bias * 0.6 + risk_hint * 0.4)))
            risk = max(0, min(2, risk))
            ts = base_ts + u * 10_000 + p * 60 + rng.randint(0, 30)
            # Add noisy social markers
            if rng.random() < 0.10:
                text = text + " http://example.com"
            if rng.random() < 0.10:
                text = "@someone " + text
            if rng.random() < 0.05:
                text = text + " email me at test@example.com"
            records.append(
                PostRecord(
                    text=text,
                    risk=risk,
                    emotion=int(emo),
                    user_id=user_id,
                    timestamp=ts,
                    lang=lang,
                    meta={"toy": True},
                )
            )

    # Shuffle overall for split logic later
    rng.shuffle(records)
    # Provide class names
    return DatasetBundle(
        train=DatasetSplit(records=[]),
        val=DatasetSplit(records=[]),
        test=DatasetSplit(records=[]),
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    ), records


def _strat_key(r: PostRecord) -> tuple[int, int]:
    return (int(r.risk), int(r.emotion))


def toy_splits(
    seed: int,
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
) -> DatasetBundle:
    bundle, records = build_toy_dataset(seed=seed, risk_classes=risk_classes, emotion_classes=emotion_classes)
    rng = random.Random(seed)

    # Simple stratification by (risk, emotion)
    buckets: dict[tuple[int, int], list[PostRecord]] = {}
    for r in records:
        buckets.setdefault(_strat_key(r), []).append(r)

    train: list[PostRecord] = []
    val: list[PostRecord] = []
    test: list[PostRecord] = []

    for _, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return DatasetBundle(
        train=DatasetSplit(records=train),
        val=DatasetSplit(records=val),
        test=DatasetSplit(records=test),
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    )
