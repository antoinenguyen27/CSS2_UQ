from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import modal

from UQ.SE_PRO.config import (
    APP_NAME,
    DEFAULT_EVAL_MODE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    HF_DATA_FILE,
    HF_DATASET,
    HF_DATASETS_CACHE,
    HF_HOME,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MODEL_CACHE_DIR,
    MODEL_ID,
    MODEL_ROOT,
    SECRET_NAME,
    TRANSFORMERS_CACHE,
    VLLM_CACHE_ROOT,
    VOLUME_NAME,
    VOLUME_ROOT,
    default_run_id,
    runtime_max_num_seqs,
)


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .env(
        {
            "HF_HOME": HF_HOME,
            "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE,
            "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
            "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .uv_pip_install(
        "datasets==4.0.0",
        "huggingface_hub>=0.34.0,<1",
        "pyarrow>=18,<22",
        "scikit-learn>=1.4,<2",
        "transformers>=4.50.0,<5",
        "vllm==0.18.0",
    )
    .add_local_python_source("UQ")
)

app = modal.App(APP_NAME)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "modal",
        "vllm",
        "transformers",
        "datasets",
        "huggingface_hub",
        "pyarrow",
        "scikit-learn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _asset_metadata_path() -> Path:
    return Path(VOLUME_ROOT) / "asset_metadata.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve_output_base(output_dir: str | None) -> Path:
    if not output_dir:
        return Path(DEFAULT_OUTPUT_DIR)
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return Path(VOLUME_ROOT) / path


def _load_runtime(requested_eval_rows: int | None) -> tuple[dict[str, Any], Path, Any, Any, int]:
    from transformers import AutoTokenizer
    from vllm import LLM

    volume.reload()
    asset_metadata_path = _asset_metadata_path()
    if not asset_metadata_path.exists():
        raise FileNotFoundError(
            "Missing /vol/asset_metadata.json. Run the MMLU-Pro asset preparation pipeline first "
            "via data_work/mmlu_pro_trace_eval/modal_app.py so the cached Gemma snapshot exists "
            "on the dedicated MMLU-Pro Modal volume."
        )

    asset_metadata = _read_json(asset_metadata_path)
    model_dir = Path(MODEL_ROOT) / MODEL_ID.replace("/", "__") / asset_metadata["model_revision"]
    max_num_seqs = runtime_max_num_seqs(requested_eval_rows)

    print(
        json.dumps(
            {
                "event": "tokenizer_load_start",
                "model_dir": str(model_dir),
                "requested_eval_rows": requested_eval_rows,
            },
            sort_keys=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        revision=asset_metadata["tokenizer_revision"],
    )
    print(
        json.dumps(
            {
                "event": "tokenizer_load_complete",
                "max_num_seqs": max_num_seqs,
            },
            sort_keys=True,
        )
    )
    print(
        json.dumps(
            {
                "event": "vllm_init_start",
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "max_num_seqs": max_num_seqs,
                "enforce_eager": True,
            },
            sort_keys=True,
        )
    )
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
    )
    print(
        json.dumps(
            {
                "event": "vllm_init_complete",
                "max_num_seqs": max_num_seqs,
            },
            sort_keys=True,
        )
    )
    return asset_metadata, model_dir, tokenizer, llm, max_num_seqs


@app.function(
    image=image,
    gpu="H200",
    cpu=6,
    memory=65_536,
    timeout=60 * 60 * 24,
    startup_timeout=60 * 30,
    secrets=[secret],
    volumes={VOLUME_ROOT: volume},
)
def run_semantic_entropy(
    eval_mode: str = DEFAULT_EVAL_MODE,
    n_samples: int = DEFAULT_N_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_eval_rows: int = 0,
    run_name: str = "",
) -> dict[str, Any]:
    import pandas as pd

    from UQ.SE_PRO.evaluation import build_results_frame, make_output_frame, summarize_partition
    from UQ.SE_PRO.sampling import (
        build_eval_partitions,
        filter_valid_rows,
        load_mmlu_pro_trace_dataframe,
        prepare_questions,
        sample_questions,
        select_eval_partition,
    )

    if eval_mode not in {"full", "split"}:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")
    if num_eval_rows < 0:
        raise ValueError(f"num_eval_rows must be non-negative, got {num_eval_rows}")

    num_eval_rows_value = num_eval_rows if num_eval_rows > 0 else None
    run_id = run_name or default_run_id(eval_mode)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    asset_metadata, _, tokenizer, llm, max_num_seqs = _load_runtime(num_eval_rows_value)

    df = load_mmlu_pro_trace_dataframe()
    valid_df, filter_drops = filter_valid_rows(df)
    if valid_df.empty:
        raise RuntimeError("No valid rows remain after filtering.")

    partitions = build_eval_partitions(valid_df, eval_mode=eval_mode, seed=seed)
    primary_split, partitions = select_eval_partition(
        partitions,
        eval_mode=eval_mode,
        num_eval_rows=num_eval_rows,
    )
    per_split_frames = []
    split_summaries: dict[str, Any] = {}
    n_questions_selected = sum(len(partition_df) for partition_df in partitions.values())

    for split_name, partition_df in partitions.items():
        questions = prepare_questions(partition_df, tokenizer)
        sampled_questions = sample_questions(
            llm=llm,
            questions=questions,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            max_num_seqs=max_num_seqs,
        )
        results_df = build_results_frame(sampled_questions, eval_split=split_name)
        per_split_frames.append(results_df)
        split_summaries[split_name] = summarize_partition(results_df)

    combined_df = make_output_frame(
        per_split_frames[0] if len(per_split_frames) == 1 else pd.concat(per_split_frames, ignore_index=True)
    )

    output_base = _resolve_output_base(output_dir)
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_parquet_path = run_dir / "semantic_entropy.parquet"
    results_csv_path = run_dir / "semantic_entropy.csv"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    combined_df.to_parquet(results_parquet_path, index=False)
    combined_df.to_csv(results_csv_path, index=False)

    primary_summary = split_summaries[primary_split]
    summary = {
        "method": "semantic_entropy",
        "benchmark": "mmlu-pro",
        "eval_mode": eval_mode,
        "primary_eval_split": primary_split,
        "run_id": run_id,
        "hf_dataset": HF_DATASET,
        "hf_data_file": HF_DATA_FILE,
        "n_valid_source_rows": int(len(valid_df)),
        "num_eval_rows_requested": num_eval_rows_value,
        "n_questions_selected": int(n_questions_selected),
        "n_questions_evaluated": primary_summary["n_questions_evaluated"],
        "n_questions_dropped": primary_summary["n_questions_dropped"],
        "drop_reason": primary_summary["drop_reasons"],
        "brier_score": primary_summary["brier_score"],
        "accuracy": primary_summary["accuracy"],
        "certainty_mean": primary_summary["certainty_mean"],
        "certainty_std": primary_summary["certainty_std"],
        "baselines": primary_summary["baselines"],
        "split_metrics": split_summaries,
        "filter_drops": filter_drops,
    }
    _write_json(summary_path, summary)

    manifest = {
        "run_id": run_id,
        "method": "semantic_entropy",
        "benchmark": "mmlu-pro",
        "eval_mode": eval_mode,
        "primary_eval_split": primary_split,
        "hf_dataset": HF_DATASET,
        "hf_data_file": HF_DATA_FILE,
        "num_eval_rows_requested": num_eval_rows_value,
        "n_valid_source_rows": int(len(valid_df)),
        "n_questions_selected": int(n_questions_selected),
        "output_dir": str(run_dir),
        "sampling_config": {
            "n_samples": n_samples,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        "asset_metadata": asset_metadata,
        "package_versions": _package_versions(),
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "results_parquet": str(results_parquet_path),
            "results_csv": str(results_csv_path),
            "summary_json": str(summary_path),
        },
    }
    _write_json(manifest_path, manifest)
    volume.commit()

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "results_parquet": str(results_parquet_path),
        "results_csv": str(results_csv_path),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_path),
        "summary": summary,
    }
