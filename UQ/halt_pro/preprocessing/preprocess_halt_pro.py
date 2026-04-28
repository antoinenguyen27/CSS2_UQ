"""
HALT-Pro preprocessing pipeline for MMLU rerun traces (example-level parquet).
"""

import numpy as np
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from pathlib import Path
import shutil
import json
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

import pandas as pd
import pyarrow.parquet as pq

HF_DATASET = "auhsoJ69/mmlu_rerun"
HF_DATA_FILE = "examples.parquet"
MAX_LEN = 192
TOP_K = 20
FEATURE_DIM = 5 + TOP_K


def compute_step_features(logprobs_step: np.ndarray):
    eps = 1e-10
    m = np.max(logprobs_step)
    probs = np.exp(logprobs_step - m)
    probs = probs / (probs.sum() + eps)

    avg_logprob = float(np.mean(logprobs_step))
    rank_proxy = float(1 + np.sum(logprobs_step[1:] > logprobs_step[0]))
    h_overall = float(-np.sum(probs * np.log(probs + eps)))

    alts_probs = probs[1:]
    alts_probs = alts_probs / (alts_probs.sum() + eps)
    h_alts = float(-np.sum(alts_probs * np.log(alts_probs + eps)))

    best_alt_lp = float(np.max(logprobs_step[1:]))
    log_sum = np.logaddexp(float(logprobs_step[0]), best_alt_lp)
    log_pc = float(logprobs_step[0]) - log_sum
    log_1_pc = best_alt_lp - log_sum
    pc = float(np.clip(np.exp(log_pc), eps, 1 - eps))
    h_dec = float(-(pc * log_pc + (1 - pc) * log_1_pc))

    feats = np.array([avg_logprob, rank_proxy, h_overall, h_alts, h_dec], dtype=np.float32)
    return feats, h_dec


def build_feature_sequence(top20_logprobs) -> np.ndarray | None:
    if top20_logprobs is None or len(top20_logprobs) == 0:
        return None

    padded = np.zeros((MAX_LEN, FEATURE_DIM), dtype=np.float32)
    prev_h_dec = 0.0
    valid_steps = 0

    for t, step in enumerate(top20_logprobs):
        if t >= MAX_LEN:
            break
        if step is None:
            continue

        step = np.array(step, dtype=np.float32)
        if step.ndim != 1 or len(step) == 0:
            continue
        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            continue
        if len(step) < TOP_K:
            step = np.pad(step, (0, TOP_K - len(step)), constant_values=-1e9)
        step = step[:TOP_K]

        stat_feats, h_dec = compute_step_features(step)
        stat_feats[4] = h_dec - prev_h_dec
        prev_h_dec = h_dec
        padded[t, :5] = stat_feats
        padded[t, 5:] = step
        valid_steps += 1

    if valid_steps == 0:
        return None
    return padded


def validate_row(row) -> tuple[bool, str]:
    if not row.get("parse_success", False):
        return False, "parse_success=False"
    if row.get("top20_token_logprobs") is None or len(row["top20_token_logprobs"]) == 0:
        return False, "empty top20_token_logprobs"
    if row.get("is_correct") is None:
        return False, "missing is_correct"
    return True, ""


def _flatten_exception_messages(exc: BaseException) -> str:
    """Collect exception + cause/context messages to detect wrapped parquet errors."""
    msgs = []
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msgs.append(str(cur))
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return " | ".join(msgs).lower()


def _is_corrupt_parquet_error(exc: BaseException) -> bool:
    msg = _flatten_exception_messages(exc)
    return (
        "corrupt snappy compressed data" in msg
        or "datasetgenerationerror" in msg
        or "an error occurred while generating the dataset" in msg
    )


def _purge_hf_cache_for_dataset(hf_dataset: str) -> None:
    """
    Remove dataset-specific cache directories so the next load truly redownloads.
    """
    ds_id = hf_dataset.lower().strip()
    needle = "datasets--" + ds_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "datasets"
    if not root.exists():
        return
    removed = 0
    for p in root.iterdir():
        name = p.name.lower()
        if needle in name or name.startswith(needle):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    if removed > 0:
        print(f"Purged {removed} Hugging Face cache directory(ies) for {hf_dataset}.")


def _load_from_converted_parquet_urls(hf_dataset: str) -> pd.DataFrame:
    """
    Fallback loader:
    - Query datasets-server parquet listing
    - Download each shard
    - Skip shards that fail parquet decode (e.g., corrupted snappy pages)
    """
    api = f"https://datasets-server.huggingface.co/parquet?dataset={quote(hf_dataset, safe='')}"
    with urlopen(api, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    shards = [x for x in payload.get("parquet_files", []) if x.get("split") == "train"]
    if not shards:
        raise RuntimeError(f"No converted parquet shards found for {hf_dataset}")

    cache_dir = Path(__file__).resolve().parents[3] / "tmp" / "halt_pro_parquet_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    skipped = []
    kept_rows = 0
    skipped_rows = 0

    print(f"Fallback: reading converted parquet shards for {hf_dataset} ({len(shards)} files).")
    for i, shard in enumerate(shards, start=1):
        url = shard["url"]
        name = shard.get("filename") or f"shard_{i:04d}.parquet"
        path = cache_dir / name
        expected_rows = int(shard.get("num_rows", 0) or 0)
        if not path.exists():
            print(f"  [{i}/{len(shards)}] Downloading {name} ...")
            urlretrieve(url, path)
        else:
            print(f"  [{i}/{len(shards)}] Using cached {name}")
        try:
            # Full decode check first (will raise on corrupt snappy blocks).
            pf = pq.ParquetFile(path)
            scanned = 0
            for rb in pf.iter_batches(batch_size=65536):
                scanned += rb.num_rows
            tbl = pq.read_table(path)
            df = tbl.to_pandas()
            frames.append(df)
            kept_rows += len(df)
            print(f"    OK rows={len(df)}")
        except Exception as e:
            # Best-effort estimate for skipped rows from parquet metadata, if readable.
            est_rows = 0
            try:
                est_rows = int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                est_rows = expected_rows
            skipped_rows += est_rows
            skipped.append((name, repr(e), est_rows))
            print(f"    SKIP {name} ({e})")

    if not frames:
        raise RuntimeError("All converted parquet shards failed to decode.")

    out = pd.concat(frames, ignore_index=True)
    print(
        "Converted parquet fallback summary: "
        f"kept_rows={kept_rows}, skipped_rows~={skipped_rows}, kept_shards={len(frames)}, skipped_shards={len(skipped)}"
    )
    for name, err, rows in skipped:
        print(f"  skipped shard: {name} (rows~={rows}) reason={err}")
    return out


def _load_train_dataframe(hf_dataset: str, data_file: str):
    """
    Load dataset and return the train split as pandas.
    If local cache shards are corrupted, force a clean redownload once.
    """
    try:
        ds = load_dataset(hf_dataset, data_files=data_file)
        return ds["train"].to_pandas()
    except (DatasetGenerationError, OSError) as e:
        if not _is_corrupt_parquet_error(e):
            raise
        print("Detected corrupted parquet cache shard. Retrying with force_redownload...")
        try:
            ds = load_dataset(hf_dataset, data_files=data_file, download_mode="force_redownload")
            return ds["train"].to_pandas()
        except (DatasetGenerationError, OSError) as e2:
            if not _is_corrupt_parquet_error(e2):
                raise
            print("Forced redownload still failed. Purging local dataset cache and retrying once...")
            _purge_hf_cache_for_dataset(hf_dataset)
            try:
                ds = load_dataset(hf_dataset, data_files=data_file, download_mode="force_redownload")
                return ds["train"].to_pandas()
            except (DatasetGenerationError, OSError) as e3:
                if not _is_corrupt_parquet_error(e3):
                    raise
                print("Cache purge retry still failed. Falling back to converted parquet shards (skip-corrupt mode).")
                return _load_from_converted_parquet_urls(hf_dataset)


def _build_example_rows_if_token_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataframe is token-step format, aggregate into example-level rows.
    Token-step signals:
      - has example_id + step_idx + top20_token_logprobs + is_correct
      - usually no parse_success column
    """
    required = {"example_id", "step_idx", "top20_token_logprobs", "is_correct"}
    if not required.issubset(set(df.columns)):
        return df
    if "parse_success" in df.columns and "top20_token_logprobs" in df.columns and "step_idx" not in df.columns:
        return df

    print("Detected token-step schema; aggregating steps into example-level sequences...")
    df2 = df.sort_values(["example_id", "step_idx"])
    grouped_rows = []
    for ex_id, g in df2.groupby("example_id", sort=False):
        grouped_rows.append(
            {
                "example_id": ex_id,
                "parse_success": True,
                "is_correct": bool(g["is_correct"].iloc[0]),
                "top20_token_logprobs": g["top20_token_logprobs"].tolist(),
            }
        )
    out = pd.DataFrame(grouped_rows)
    print(f"Aggregated {len(df2)} token-steps into {len(out)} examples.")
    return out


def preprocess(hf_dataset: str = HF_DATASET, data_file: str = HF_DATA_FILE) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading dataset: {hf_dataset} ({data_file})")
    df = _load_train_dataframe(hf_dataset, data_file)
    df = _build_example_rows_if_token_steps(df)
    print(f"Total rows: {len(df)}")

    features_list = []
    labels_list = []
    skipped = {}

    for _, row in df.iterrows():
        ok, reason = validate_row(row)
        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        features = build_feature_sequence(row["top20_token_logprobs"])
        if features is None:
            skipped["bad feature sequence"] = skipped.get("bad feature sequence", 0) + 1
            continue
        features_list.append(features)
        labels_list.append(float(row["is_correct"]))

    if not features_list:
        raise RuntimeError("No valid rows found.")

    x = np.stack(features_list, axis=0)
    y = np.array(labels_list, dtype=np.int8)
    print(f"Rows kept: {len(x)} | Rows skipped: {sum(skipped.values())}")
    for k, v in skipped.items():
        if v > 0:
            print(f"  {k}: {v}")
    print(f"Label balance — correct: {int(y.sum())} | incorrect: {len(y) - int(y.sum())}")
    print(f"Feature matrix shape: {x.shape}")
    return x, y

