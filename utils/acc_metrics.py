import argparse
import json
from pathlib import Path

import numpy as np


def _load_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "data" in arr:
            return arr["data"]
        raise ValueError(f"NPZ file {path} must contain a 'data' array")
    return arr


def _lat_weights(lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))
    weights = weights / np.mean(weights)
    return weights


def anomaly_correlation(
    pred: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    lat: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if pred.shape != truth.shape:
        raise ValueError(f"pred shape {pred.shape} does not match truth {truth.shape}")

    if climatology.shape != truth.shape[1:]:
        raise ValueError(
            f"climatology shape {climatology.shape} must match spatial shape {truth.shape[1:]}"
        )

    pred_anom = pred - climatology
    truth_anom = truth - climatology

    if lat is not None:
        weights = _lat_weights(lat)
        weights = weights.reshape((1, -1) + (1,) * (pred_anom.ndim - 2))
        pred_anom = pred_anom * weights
        truth_anom = truth_anom * weights

    num = np.sum(pred_anom * truth_anom, axis=tuple(range(1, pred_anom.ndim)))
    denom = np.sqrt(
        np.sum(pred_anom ** 2, axis=tuple(range(1, pred_anom.ndim)))
        * np.sum(truth_anom ** 2, axis=tuple(range(1, pred_anom.ndim)))
    )
    acc = np.where(denom > 0, num / denom, 0.0)
    return acc, float(np.mean(acc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute anomaly correlation coefficient (ACC).")
    parser.add_argument("--pred", required=True, help="Path to predictions (.npy or .npz with 'data').")
    parser.add_argument("--truth", required=True, help="Path to ground truth (.npy or .npz with 'data').")
    parser.add_argument("--climatology", help="Path to climatology array (.npy or .npz with 'data').")
    parser.add_argument("--lat", help="Optional latitude array (.npy) for cosine weighting.")
    parser.add_argument("--output", help="Optional path to write JSON summary.")
    args = parser.parse_args()

    pred = _load_array(args.pred)
    truth = _load_array(args.truth)

    if args.climatology:
        climatology = _load_array(args.climatology)
    else:
        climatology = np.mean(truth, axis=0)

    lat = _load_array(args.lat) if args.lat else None
    acc_per_step, acc_mean = anomaly_correlation(pred, truth, climatology, lat=lat)

    summary = {
        "acc_mean": acc_mean,
        "acc_per_step": acc_per_step.tolist(),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
