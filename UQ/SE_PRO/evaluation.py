from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from UQ.SE_PRO.config import ANSWER_LETTERS, MIN_VALID_SAMPLES, answer_letters_for_choices


def compute_brier_score(y_true: np.ndarray, predicted_prob: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(np.mean((y_true - predicted_prob) ** 2))


def compute_accuracy(y_true: np.ndarray, predicted_prob: np.ndarray, threshold: float = 0.5) -> float | None:
    if len(y_true) == 0:
        return None
    predicted_label = (predicted_prob >= threshold).astype(np.int8)
    return float(np.mean(predicted_label == y_true))


def semantic_entropy_from_answers(
    valid_answers: list[str],
    answer_letters: Sequence[str],
) -> dict[str, Any]:
    active_letters = tuple(answer_letters)
    counts = Counter(answer for answer in valid_answers if answer in active_letters)
    n_valid = len(valid_answers)

    if n_valid == 0:
        probabilities = {letter: 0.0 for letter in ANSWER_LETTERS}
        return {
            "n_valid_samples": 0,
            "counts": {letter: 0 for letter in ANSWER_LETTERS},
            "probabilities": probabilities,
            "entropy": None,
            "normalized_entropy": None,
            "certainty": None,
        }

    probabilities_array = np.array(
        [counts.get(letter, 0) / n_valid for letter in active_letters],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_terms = np.nan_to_num(probabilities_array * np.log(probabilities_array), nan=0.0)
    entropy = float(-safe_terms.sum())
    normalized_entropy = float(entropy / np.log(len(active_letters)))
    certainty = float(1.0 - normalized_entropy)

    probabilities = {letter: 0.0 for letter in ANSWER_LETTERS}
    for idx, letter in enumerate(active_letters):
        probabilities[letter] = float(probabilities_array[idx])

    return {
        "n_valid_samples": n_valid,
        "counts": {letter: counts.get(letter, 0) for letter in ANSWER_LETTERS},
        "probabilities": probabilities,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "certainty": certainty,
    }


def build_results_frame(question_samples: list[dict[str, Any]], eval_split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in question_samples:
        choices = list(question["choices"])
        answer_letters = answer_letters_for_choices(choices)
        stats = semantic_entropy_from_answers(question["valid_answers"], answer_letters=answer_letters)
        dropped = stats["n_valid_samples"] < MIN_VALID_SAMPLES
        row = {
            "example_id": question["example_id"],
            "subject": question["subject"],
            "source_split": question["source_split"],
            "eval_split": eval_split,
            "gold_answer": question["gold_answer"],
            "is_correct": int(question["is_correct"]),
            "n_choices": len(choices),
            "choices": choices,
            "n_valid_samples": stats["n_valid_samples"],
            "entropy": None if dropped else stats["entropy"],
            "normalized_entropy": None if dropped else stats["normalized_entropy"],
            "certainty": None if dropped else stats["certainty"],
            "missing_sample_count": question["missing_sample_count"],
            "drop_reason": question["drop_reason"] if dropped else "",
            "sampled_token_logprobs": question.get("sampled_token_logprobs"),
        }
        for letter in ANSWER_LETTERS:
            row[f"n_{letter}"] = stats["counts"][letter]
        for letter in ANSWER_LETTERS:
            row[f"p_{letter}"] = stats["probabilities"][letter]
        rows.append(row)
    return pd.DataFrame(rows)


def majority_class_baseline(y_true: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"brier_score": None, "accuracy": None, "predicted_probability": None}
    majority_label = int(np.mean(y_true) >= 0.5)
    predicted = np.full_like(y_true, majority_label, dtype=np.float64)
    return {
        "brier_score": compute_brier_score(y_true, predicted),
        "accuracy": compute_accuracy(y_true, predicted),
        "predicted_probability": float(majority_label),
    }


def token_logprob_baseline(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"brier_score": None, "accuracy": None}

    avg_logprobs: list[float] = []
    labels: list[int] = []
    for row in df.itertuples(index=False):
        logprobs = getattr(row, "sampled_token_logprobs", None)
        if logprobs is None:
            continue
        array = np.asarray(logprobs, dtype=np.float32)
        if array.size == 0:
            continue
        array = np.nan_to_num(array, nan=-100.0, posinf=-100.0, neginf=-100.0)
        avg_logprobs.append(float(array.mean()))
        labels.append(int(row.is_correct))

    if not avg_logprobs:
        return {"brier_score": None, "accuracy": None}

    avg_array = np.asarray(avg_logprobs, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    min_value = float(avg_array.min())
    max_value = float(avg_array.max())
    if max_value > min_value:
        predicted = (avg_array - min_value) / (max_value - min_value)
    else:
        predicted = np.full_like(avg_array, 0.5, dtype=np.float64)

    return {
        "brier_score": compute_brier_score(label_array, predicted),
        "accuracy": compute_accuracy(label_array, predicted),
    }


def summarize_partition(results_df: pd.DataFrame) -> dict[str, Any]:
    evaluated_df = results_df[results_df["drop_reason"] == ""].copy()
    y_true = evaluated_df["is_correct"].to_numpy(dtype=np.int8)
    certainty = evaluated_df["certainty"].to_numpy(dtype=np.float64)

    return {
        "n_questions_total": int(len(results_df)),
        "n_questions_evaluated": int(len(evaluated_df)),
        "n_questions_dropped": int(len(results_df) - len(evaluated_df)),
        "drop_reasons": results_df[results_df["drop_reason"] != ""]["drop_reason"].value_counts().to_dict(),
        "brier_score": compute_brier_score(y_true, certainty),
        "accuracy": compute_accuracy(y_true, certainty),
        "certainty_mean": float(np.mean(certainty)) if len(certainty) else None,
        "certainty_std": float(np.std(certainty)) if len(certainty) else None,
        "baselines": {
            "majority_class": majority_class_baseline(y_true),
            "token_logprob": token_logprob_baseline(evaluated_df),
        },
    }


def make_output_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    output_df = results_df.copy()
    if "sampled_token_logprobs" in output_df.columns:
        output_df = output_df.drop(columns=["sampled_token_logprobs"])
    return output_df
