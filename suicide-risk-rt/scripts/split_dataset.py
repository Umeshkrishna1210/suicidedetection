from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _infer_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in (".xlsx", ".xls"):
        return "xlsx"
    if suf in (".jsonl", ".json"):
        return "jsonl"
    raise SystemExit(f"Unsupported input extension '{suf}'. Use .csv/.xlsx/.jsonl")


def _read(path: Path, *, encoding: str | None = None, usecols: list[str] | None = None):
    fmt = _infer_format(path)
    if fmt == "csv":
        read_kwargs = {
            "engine": "python",
            "on_bad_lines": "skip",
        }
        if usecols is not None:
            read_kwargs["usecols"] = usecols

        # Many CSVs exported from Excel/Kaggle are not UTF-8.
        # Try a few common encodings before failing.
        if encoding and str(encoding).strip():
            return pd.read_csv(path, encoding=str(encoding).strip(), **read_kwargs)

        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err: Exception | None = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **read_kwargs)
            except UnicodeDecodeError as e:
                last_err = e
        if last_err is not None:
            raise last_err
        return pd.read_csv(path, **read_kwargs)
    if fmt == "xlsx":
        return pd.read_excel(path, usecols=usecols)
    if fmt == "jsonl":
        df = pd.read_json(path, lines=True)
        if usecols is not None:
            missing = [c for c in usecols if c not in df.columns]
            if missing:
                raise SystemExit(f"Missing required columns in jsonl: {missing}")
            return df[usecols]
        return df
    raise SystemExit("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a single dataset file into train/val/test (stratified by label).")
    ap.add_argument("--in", dest="inp", required=True, help="Input file: .csv/.xlsx/.jsonl")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--encoding",
        default="",
        help="Optional CSV encoding (e.g., cp1252). If set, overrides auto-detection attempts.",
    )
    ap.add_argument(
        "--allowed-labels",
        nargs="*",
        default=None,
        help="Optional list of allowed label values; other rows will be dropped (case-insensitive).",
    )
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-cols", nargs="*", default=None, help="Optional columns to drop (e.g., id, index)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_col = str(args.text_field)
    label_col = str(args.label_field)
    df = _read(inp, encoding=str(args.encoding).strip() or None, usecols=[text_col, label_col])

    if args.drop_cols:
        for c in args.drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    if text_col not in df.columns or label_col not in df.columns:
        raise SystemExit(f"Missing required columns. Found={list(df.columns)} required='{text_col}', '{label_col}'")

    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    # Basic cleanup
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].str.strip()
    df[label_col] = df[label_col].str.strip()
    df = df[df[text_col].str.len() > 0]
    df = df[df[label_col].str.len() > 0]

    # Drop malformed labels (common when CSV has stray commas/rows)
    if args.allowed_labels is not None and len(args.allowed_labels) > 0:
        allowed = {str(x).strip().lower() for x in args.allowed_labels}
        df = df[df[label_col].str.strip().str.lower().isin(allowed)]
    else:
        # Auto-filter for the common suicide dataset case.
        uniq = set(df[label_col].str.strip().str.lower().unique().tolist())
        if "suicide" in uniq and "non-suicide" in uniq:
            df = df[df[label_col].str.strip().str.lower().isin({"suicide", "non-suicide"})]

    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise SystemExit("Invalid ratios: require val_ratio>=0, test_ratio>=0, val_ratio+test_ratio<1")

    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise SystemExit("scikit-learn is required for stratified splitting") from e

    strat = df[label_col]
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=int(args.seed),
        shuffle=True,
        stratify=strat,
    )

    if len(temp_df) == 0:
        raise SystemExit("Not enough data to create val/test splits")

    rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    if rel_test == 0.0:
        val_df = temp_df
        test_df = temp_df.iloc[:0].copy()
    else:
        strat2 = temp_df[label_col]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test,
            random_state=int(args.seed),
            shuffle=True,
            stratify=strat2,
        )

    # Keep only required cols by default
    keep_cols = [text_col, label_col]
    extra_cols = [c for c in df.columns if c not in keep_cols]
    # Keep extra cols if they exist (user_id/timestamp/lang), otherwise fine
    keep = keep_cols + extra_cols

    train_df[keep].to_csv(out_dir / "train.csv", index=False)
    val_df[keep].to_csv(out_dir / "val.csv", index=False)
    test_df[keep].to_csv(out_dir / "test.csv", index=False)

    def pct(x: int, total: int) -> str:
        return f"{(100.0 * x / max(1,total)):.2f}%"

    total = len(df)
    print("OK split")
    print(f"  total: {total}")
    print(f"  train: {len(train_df)} ({pct(len(train_df), total)})")
    print(f"  val:   {len(val_df)} ({pct(len(val_df), total)})")
    print(f"  test:  {len(test_df)} ({pct(len(test_df), total)})")
    print("  label distribution (total):")
    print(df[label_col].value_counts(dropna=False).head(20).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
