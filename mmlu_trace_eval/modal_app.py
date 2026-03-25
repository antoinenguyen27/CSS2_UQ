from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import modal

from .batching import PreparedRequest
from .config import (
    APP_NAME,
    CACHE_ROOT,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_ROOT,
    DEFAULT_EVALUATOR_CPU,
    DEFAULT_EVALUATOR_MEMORY_MB,
    DEFAULT_PREPARE_CPU,
    DEFAULT_PREPARE_MEMORY_MB,
    DecodeConfig,
    HF_DATASETS_CACHE,
    HF_HOME,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MODEL_CACHE_DIR,
    MODEL_ID,
    MODEL_ROOT,
    PROMPT_VARIANT,
    RUNS_ROOT,
    SECRET_NAME,
    SHARD_SIZE,
    SUPPORTED_SPLITS,
    SYSTEM_PROMPT,
    TRANSFORMERS_CACHE,
    USER_PROMPT_TEMPLATE,
    VLLM_CACHE_ROOT,
    VOLUME_NAME,
    VOLUME_ROOT,
    default_run_id,
    runtime_max_num_seqs,
)
from .dataset import build_example_id, filter_examples, load_materialized_split, normalized_dataset_dir
from .prompting import build_messages, count_prompt_tokens, render_prompt
from .schema import (
    build_example_record,
    build_token_step_records,
    compute_token_surfaces,
    normalize_topk,
    segment_token_surfaces,
    validate_example_record,
)
from .storage import (
    MetricsAccumulator,
    build_manifest,
    compact_shards,
    ensure_dir,
    iter_example_records_from_shards,
    list_completed_example_ids,
    read_json,
    write_json,
    write_shard,
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
        "transformers>=4.50.0,<5",
        "vllm==0.18.0",
    )
    .add_local_python_source("mmlu_trace_eval")
)

app = modal.App(APP_NAME)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("modal", "vllm", "transformers", "datasets", "huggingface_hub", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _asset_metadata_path() -> Path:
    return Path(VOLUME_ROOT) / "asset_metadata.json"


def _model_dir(model_revision: str) -> Path:
    return Path(MODEL_ROOT) / MODEL_ID.replace("/", "__") / model_revision


def _dataset_ready_path(dataset_revision: str) -> Path:
    return normalized_dataset_dir(DATASET_ROOT, dataset_revision) / ".ready"


def _model_ready_path(model_revision: str) -> Path:
    return _model_dir(model_revision) / ".ready"


@app.function(
    image=image,
    cpu=DEFAULT_PREPARE_CPU,
    memory=DEFAULT_PREPARE_MEMORY_MB,
    timeout=60 * 60 * 2,
    startup_timeout=60 * 20,
    secrets=[secret],
    volumes={VOLUME_ROOT: volume},
)
def prepare_assets() -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    ensure_dir(Path(MODEL_ROOT))
    ensure_dir(Path(DATASET_ROOT))
    ensure_dir(Path(CACHE_ROOT))

    api = HfApi(token=os.environ["HF_TOKEN"])
    model_revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha

    model_dir = _model_dir(model_revision)
    model_ready = _model_ready_path(model_revision)
    if not model_ready.exists():
        ensure_dir(model_dir)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=model_revision,
            local_dir=str(model_dir),
            token=os.environ["HF_TOKEN"],
        )
        model_ready.write_text(model_revision + "\n")
        volume.commit()

    dataset_dir = normalized_dataset_dir(DATASET_ROOT, dataset_revision)
    dataset_ready = _dataset_ready_path(dataset_revision)
    if not dataset_ready.exists():
        ensure_dir(dataset_dir)
        for split in SUPPORTED_SPLITS:
            dataset = load_dataset(
                DATASET_ID,
                DATASET_CONFIG,
                split=split,
                revision=dataset_revision,
                cache_dir=HF_DATASETS_CACHE,
            )
            records: list[dict[str, Any]] = []
            for question_idx, row in enumerate(dataset):
                records.append(
                    {
                        "example_id": build_example_id(split, row["subject"], question_idx),
                        "question_idx": question_idx,
                        "subject": row["subject"],
                        "question": row["question"],
                        "choices": list(row["choices"]),
                        "gold_answer": "ABCD"[int(row["answer"])],
                    }
                )
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(records), dataset_dir / f"{split}.parquet")
        dataset_ready.write_text(dataset_revision + "\n")
        volume.commit()

    asset_metadata = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
    }
    write_json(_asset_metadata_path(), asset_metadata)
    volume.commit()
    return asset_metadata


@app.cls(
    image=image,
    gpu="H200",
    cpu=DEFAULT_EVALUATOR_CPU,
    memory=DEFAULT_EVALUATOR_MEMORY_MB,
    timeout=60 * 60 * 24,
    startup_timeout=60 * 30,
    secrets=[secret],
    volumes={VOLUME_ROOT: volume},
)
class Evaluator:
    requested_split: str = modal.parameter(default="test")

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        volume.reload()
        asset_metadata = read_json(_asset_metadata_path())
        self.asset_metadata = asset_metadata
        self.model_dir = _model_dir(asset_metadata["model_revision"])
        self.max_num_seqs = runtime_max_num_seqs(self.requested_split)

        print(
            json.dumps(
                {
                    "event": "tokenizer_load_start",
                    "requested_split": self.requested_split,
                    "model_dir": str(self.model_dir),
                },
                sort_keys=True,
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            revision=asset_metadata["tokenizer_revision"],
        )
        print(
            json.dumps(
                {
                    "event": "tokenizer_load_complete",
                    "requested_split": self.requested_split,
                    "max_num_seqs": self.max_num_seqs,
                },
                sort_keys=True,
            )
        )
        print(
            json.dumps(
                {
                    "event": "vllm_init_start",
                    "requested_split": self.requested_split,
                    "max_model_len": MAX_MODEL_LEN,
                    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                    "max_num_seqs": self.max_num_seqs,
                    "enforce_eager": True,
                    "model_dir": str(self.model_dir),
                },
                sort_keys=True,
            )
        )
        self.llm = LLM(
            model=str(self.model_dir),
            tokenizer=str(self.model_dir),
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 0},
        )
        print(
            json.dumps(
                {
                    "event": "vllm_init_complete",
                    "requested_split": self.requested_split,
                    "max_num_seqs": self.max_num_seqs,
                    "enforce_eager": True,
                },
                sort_keys=True,
            )
        )

    @modal.method()
    def run(
        self,
        *,
        split: str,
        limit: int | None,
        subject: str | None,
        run_id: str,
        resume: bool = True,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        volume.reload()

        dataset_revision = self.asset_metadata["dataset_revision"]
        examples = load_materialized_split(DATASET_ROOT, dataset_revision, split)
        examples = filter_examples(examples, subject=subject, limit=limit)
        if not examples:
            raise ValueError("No examples remain after applying split, subject, and limit filters")

        run_dir = Path(RUNS_ROOT) / run_id
        shards_dir = run_dir / "shards"
        if run_dir.exists() and not resume and any(run_dir.iterdir()):
            raise ValueError(f"Run directory already exists and resume is false: {run_dir}")
        ensure_dir(shards_dir)

        completed_example_ids = list_completed_example_ids(run_dir) if resume else set()
        pending_examples = [row for row in examples if row["example_id"] not in completed_example_ids]

        existing_manifest_path = run_dir / "manifest.json"
        existing_manifest = read_json(existing_manifest_path) if resume and existing_manifest_path.exists() else None

        prompt_hash = hashlib.sha256(
            f"{SYSTEM_PROMPT}\n{USER_PROMPT_TEMPLATE}\n{self.tokenizer.chat_template or ''}".encode("utf-8")
        ).hexdigest()
        decode_config = DecodeConfig()
        decode_config_json = decode_config.to_json()
        started_at_utc = (
            existing_manifest["started_at_utc"]
            if existing_manifest and existing_manifest.get("started_at_utc")
            else datetime.now(timezone.utc).isoformat()
        )
        shard_inventory: list[dict[str, Any]] = []
        metrics = MetricsAccumulator()

        if resume:
            for metadata_path in sorted(shards_dir.glob("meta-*.json")):
                shard_inventory.append(read_json(metadata_path))
            for example_record in iter_example_records_from_shards(run_dir):
                metrics.update(example_record)

        manifest = build_manifest(
            run_id=run_id,
            split=split,
            subject=subject,
            requested_limit=limit,
            total_examples=len(examples),
            asset_metadata=self.asset_metadata,
            prompt_variant=PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            decode_config_json=decode_config_json,
            package_versions=_package_versions(),
            shard_inventory=shard_inventory,
            started_at_utc=started_at_utc,
            finished_at_utc=None,
        )
        write_json(run_dir / "manifest.json", manifest)

        prepared_requests = self._prepare_requests(pending_examples)

        sampling_params = SamplingParams(
            temperature=decode_config.temperature,
            top_p=decode_config.top_p,
            max_tokens=decode_config.max_tokens,
            logprobs=decode_config.logprobs,
            stop=list(decode_config.stop),
            include_stop_str_in_output=decode_config.include_stop_str_in_output,
            seed=decode_config.seed,
            n=decode_config.n,
        )

        # Phase 1: Generate all outputs in a single call.
        # vLLM's internal continuous batching keeps the GPU fully utilized
        # without inter-batch CPU idle gaps.
        all_prompts = [r.prompt_text for r in prepared_requests]
        all_outputs = self.llm.generate(all_prompts, sampling_params)

        # Phase 2: Convert records and write shards.
        # Generation is complete so this CPU work no longer starves the GPU.
        next_shard_index = len(shard_inventory)
        shard_example_records: list[dict[str, Any]] = []
        shard_token_step_records: list[dict[str, Any]] = []

        for request, output in zip(prepared_requests, all_outputs, strict=True):
            example_record = self._convert_output_to_record(
                request=request,
                output=output,
                run_id=run_id,
                split=split,
                decode_config_json=decode_config_json,
            )
            validate_example_record(example_record)
            token_step_records = build_token_step_records(example_record)
            shard_example_records.append(example_record)
            shard_token_step_records.extend(token_step_records)
            metrics.update(example_record)

            if len(shard_example_records) >= SHARD_SIZE:
                metadata = write_shard(run_dir, next_shard_index, shard_example_records, shard_token_step_records)
                shard_inventory.append(metadata)
                write_json(
                    run_dir / "manifest.json",
                    build_manifest(
                        run_id=run_id,
                        split=split,
                        subject=subject,
                        requested_limit=limit,
                        total_examples=len(examples),
                        asset_metadata=self.asset_metadata,
                        prompt_variant=PROMPT_VARIANT,
                        prompt_hash=prompt_hash,
                        decode_config_json=decode_config_json,
                        package_versions=_package_versions(),
                        shard_inventory=shard_inventory,
                        started_at_utc=started_at_utc,
                        finished_at_utc=None,
                    ),
                )
                volume.commit()
                next_shard_index += 1
                shard_example_records = []
                shard_token_step_records = []

        if shard_example_records:
            metadata = write_shard(run_dir, next_shard_index, shard_example_records, shard_token_step_records)
            shard_inventory.append(metadata)
            volume.commit()

        compact_shards(run_dir, "examples", "examples.parquet")
        compact_shards(run_dir, "token_steps", "token_steps.parquet")
        metrics_payload = metrics.to_dict()
        write_json(run_dir / "metrics.json", metrics_payload)

        finished_at_utc = datetime.now(timezone.utc).isoformat()
        final_manifest = build_manifest(
            run_id=run_id,
            split=split,
            subject=subject,
            requested_limit=limit,
            total_examples=len(examples),
            asset_metadata=self.asset_metadata,
            prompt_variant=PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            decode_config_json=decode_config_json,
            package_versions=_package_versions(),
            shard_inventory=shard_inventory,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
        )
        write_json(run_dir / "manifest.json", final_manifest)
        volume.commit()

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "examples_path": str(run_dir / "examples.parquet"),
            "token_steps_path": str(run_dir / "token_steps.parquet"),
            "manifest_path": str(run_dir / "manifest.json"),
            "metrics_path": str(run_dir / "metrics.json"),
            "metrics": metrics_payload,
        }

    def _prepare_requests(self, examples: list[dict[str, Any]]) -> list[PreparedRequest]:
        requests: list[PreparedRequest] = []
        for example in examples:
            messages = build_messages(example)
            prompt_text = render_prompt(self.tokenizer, messages)
            prompt_tokens = count_prompt_tokens(self.tokenizer, messages)
            requests.append(
                PreparedRequest(
                    example=example,
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                )
            )
        return requests

    def _convert_output_to_record(
        self,
        *,
        request: PreparedRequest,
        output: Any,
        run_id: str,
        split: str,
        decode_config_json: str,
    ) -> dict[str, Any]:
        generated = output.outputs[0]
        sampled_token_ids = [int(token_id) for token_id in generated.token_ids]
        sampled_token_texts = [
            self.tokenizer.convert_ids_to_tokens(int(token_id))
            for token_id in sampled_token_ids
        ]

        top20_token_ids: list[list[int]] = []
        top20_token_texts: list[list[str]] = []
        top20_token_logprobs: list[list[float]] = []
        sampled_token_logprobs: list[float] = []
        sampled_token_ranks: list[int] = []
        cumulative_logprobs: list[float] = []
        running_logprob = 0.0

        step_logprob_iterable = generated.logprobs or [None] * len(sampled_token_ids)
        for sampled_token_id, step_logprobs in zip(sampled_token_ids, step_logprob_iterable, strict=True):
            normalized = normalize_topk(self.tokenizer, step_logprobs, sampled_token_id)
            sampled_token_logprobs.append(normalized.sampled_logprob)
            sampled_token_ranks.append(normalized.sampled_rank)
            running_logprob += normalized.sampled_logprob
            cumulative_logprobs.append(running_logprob)
            top20_token_ids.append(normalized.token_ids)
            top20_token_texts.append(normalized.token_texts)
            top20_token_logprobs.append(normalized.logprobs)

        completion_text = generated.text
        token_surfaces = compute_token_surfaces(self.tokenizer, sampled_token_ids)
        segment_ids, segment_names = segment_token_surfaces(completion_text, token_surfaces)

        return build_example_record(
            run_id=run_id,
            split=split,
            prompt_variant=PROMPT_VARIANT,
            decode_config_json=decode_config_json,
            model_name=self.asset_metadata["model_id"],
            model_revision=self.asset_metadata["model_revision"],
            tokenizer_revision=self.asset_metadata["tokenizer_revision"],
            dataset_name=self.asset_metadata["dataset_id"],
            dataset_revision=self.asset_metadata["dataset_revision"],
            prompt_text=request.prompt_text,
            example=request.example,
            completion_text=completion_text,
            sampled_token_ids=sampled_token_ids,
            sampled_token_texts=sampled_token_texts,
            sampled_token_logprobs=sampled_token_logprobs,
            sampled_token_ranks=sampled_token_ranks,
            segment_ids=segment_ids,
            segment_names=segment_names,
            cumulative_logprobs=cumulative_logprobs,
            top20_token_ids=top20_token_ids,
            top20_token_texts=top20_token_texts,
            top20_token_logprobs=top20_token_logprobs,
        )


@app.local_entrypoint()
def main(
    split: str = "test",
    limit: int = 0,
    subject: str = "",
    run_name: str = "",
    resume: bool = True,
) -> None:
    limit_value = limit if limit > 0 else None
    subject_value = subject or None

    asset_metadata = prepare_assets.remote()
    run_id = run_name or default_run_id(split=split, subject=subject_value)
    result = Evaluator(requested_split=split).run.remote(
        split=split,
        limit=limit_value,
        subject=subject_value,
        run_id=run_id,
        resume=resume,
    )
    print(json.dumps({"asset_metadata": asset_metadata, "result": result}, indent=2, sort_keys=True))
