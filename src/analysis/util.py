import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F


def parse_filename_params(filename: str) -> dict:
    """Parse [key=value] parameters from an eval result filename."""
    name = filename.replace(".json", "")
    pattern = r"\[([^=]+)=([^\]]+)\]"
    params = {}
    for key, value in re.findall(pattern, name):
        if value.replace(".", "").replace("-", "").isdigit():
            try:
                params[key] = float(value) if "." in value else int(value)
            except ValueError:
                params[key] = value
        else:
            params[key] = value
    return params


def extract_models_from_filename(filename: str) -> tuple:
    """Extract and sanitize retriever/reranker model names from filename."""
    params = parse_filename_params(filename)
    retriever = params.get("biencoder_model_path")
    reranker = params.get("reranker_model_path")
    reranker_type = params.get("reranker_type")

    def _sanitize(name):
        if name is None:
            return None
        return str(name).replace("/", "--").replace(" ", "_")

    return (
        _sanitize(retriever),
        _sanitize(reranker),
        (str(reranker_type) if reranker_type is not None else None),
    )


def load_calibration(calibration_file: str | None) -> dict:
    """Load calibration params from JSON. Returns empty dict if file is None."""
    if calibration_file is None:
        return {}
    with open(calibration_file) as f:
        return json.load(f)


def get_calibration_params(
    reranker_model_name: str | None, calibration_data: dict
) -> dict | None:
    """Look up calibration params for a model by its sanitized name.

    Sanitized names use '--' as separator (e.g. 'redis--langcache-reranker-v1-bce'),
    while calibration keys use only the final path component ('langcache-reranker-v1-bce').

    Handles both flat {param: value} and nested {version: {param: value}} structures
    produced by compute_calibration.py. For the nested form, the latest version is used.
    """
    if not reranker_model_name or not calibration_data:
        return None
    calib_key = reranker_model_name.split("--")[-1]
    model_entry = calibration_data.get(calib_key)
    if not model_entry:
        return None
    first_val = next(iter(model_entry.values()))
    if isinstance(first_val, dict):
        latest_version = sorted(model_entry.keys())[-1]
        return model_entry[latest_version]
    return model_entry


def normalize_scores_softmax(scores: list) -> list:
    """Apply softmax normalization across a list of scores."""
    if not scores:
        return []
    try:
        tensor_scores = torch.tensor(scores, dtype=torch.float32)
        probs = F.softmax(tensor_scores, dim=0).cpu().numpy()
        return probs.tolist()
    except Exception:
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))
        return (exp_scores / np.sum(exp_scores)).tolist()


def apply_calibrated_sigmoid(logit: float, calib: dict, method: str) -> float:
    """Apply temperature or Platt calibration to a single logit, returning a probability in [0, 1]."""
    if method == "temperature":
        scaled = logit / calib["temperature"]
    else:  # platt
        scaled = calib["platt_a"] * logit + calib["platt_b"]
    return float(torch.sigmoid(torch.tensor(scaled, dtype=torch.float32)).item())


def normalize_reranker_scores(
    ranked_scores: list,
    reranker_type: str | None,
    calib_params: dict | None = None,
    calibration_method: str = "temperature",
) -> list:
    """Normalize raw reranker logits to [0, 1] probabilities.

    - ColBERT: always softmax across all candidates, never calibrated.
    - All others (BCE, MNRL, etc.): sigmoid, calibrated if calib_params provided.
    """
    if not ranked_scores:
        return ranked_scores

    reranker_type_l = str(reranker_type).lower() if reranker_type is not None else None

    if reranker_type_l == "colbert":
        return normalize_scores_softmax(ranked_scores)

    if calib_params is not None:
        try:
            return [
                apply_calibrated_sigmoid(float(x), calib_params, calibration_method)
                for x in ranked_scores
            ]
        except Exception:
            pass

    # plain sigmoid fallback
    try:
        return [
            float(torch.sigmoid(torch.tensor(float(x), dtype=torch.float32)).item())
            for x in ranked_scores
        ]
    except Exception:
        try:
            from scipy.special import expit

            return expit(np.array(ranked_scores, dtype=float)).tolist()
        except Exception:
            return list(ranked_scores)


def nan_to_none(x: float) -> float | None:
    """Convert NaN/Inf floats to None for JSON serialization."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x
