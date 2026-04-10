from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router-probe outputs.")
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None, help="Exclusive upper bound.")
    return parser.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(m, 1e-12, None))))
    kl_qm = np.sum(q * (np.log(np.clip(q, 1e-12, None)) - np.log(np.clip(m, 1e-12, None))))
    return float(0.5 * (kl_pm + kl_qm))


def load_router_probs(npz_path: str) -> np.ndarray:
    with np.load(npz_path) as data:
        return data["router_probs_mean"]


def select_layers(array: np.ndarray, layer_start: int | None, layer_end: int | None) -> np.ndarray:
    start = 0 if layer_start is None else layer_start
    end = array.shape[0] if layer_end is None else layer_end
    return array[start:end]


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if df.empty:
        raise ValueError("results.csv is empty")

    summary_rows = []
    layerwise_records = []

    for clip_id, clip_df in df.groupby("clip_id"):
        if "aligned" not in set(clip_df["variant"]):
            continue
        aligned_row = clip_df.loc[clip_df["variant"] == "aligned"].iloc[0]
        aligned_probs = select_layers(load_router_probs(aligned_row["yes_feature_path"]), args.layer_start, args.layer_end)

        for _, other_row in clip_df.iterrows():
            if other_row["variant"] == "aligned":
                continue
            other_probs = select_layers(load_router_probs(other_row["yes_feature_path"]), args.layer_start, args.layer_end)
            layerwise_js = [js_divergence(aligned_probs[layer], other_probs[layer]) for layer in range(aligned_probs.shape[0])]
            mean_js = float(np.mean(layerwise_js))
            summary_rows.append(
                {
                    "clip_id": clip_id,
                    "aligned_variant": "aligned",
                    "other_variant": other_row["variant"],
                    "other_alignment_label": int(other_row["alignment_label"]),
                    "other_perturbation": other_row["perturbation"],
                    "aligned_margin": float(aligned_row["yes_minus_no_margin"]),
                    "other_margin": float(other_row["yes_minus_no_margin"]),
                    "delta_margin": float(aligned_row["yes_minus_no_margin"] - other_row["yes_minus_no_margin"]),
                    "mean_js_yes_answer": mean_js,
                }
            )
            for layer_index, layer_js in enumerate(layerwise_js):
                layerwise_records.append(
                    {
                        "clip_id": clip_id,
                        "other_variant": other_row["variant"],
                        "other_alignment_label": int(other_row["alignment_label"]),
                        "other_perturbation": other_row["perturbation"],
                        "layer_index": layer_index,
                        "js_yes_answer": float(layer_js),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    layerwise_df = pd.DataFrame(layerwise_records)
    summary_df.to_csv(args.output_dir / "summary_by_clip.csv", index=False)
    layerwise_df.to_csv(args.output_dir / "layerwise_js_shift_vs_aligned.csv", index=False)

    aggregate = (
        layerwise_df.groupby(["other_perturbation", "layer_index"], as_index=False)["js_yes_answer"]
        .mean()
        .sort_values(["other_perturbation", "layer_index"])
    )
    aggregate.to_csv(args.output_dir / "layerwise_js_aggregate.csv", index=False)

    plt.figure(figsize=(10, 5))
    for perturbation, part in aggregate.groupby("other_perturbation"):
        plt.plot(part["layer_index"], part["js_yes_answer"], label=perturbation)
    plt.xlabel("Layer index")
    plt.ylabel("Mean JS divergence on `yes` answer router distribution")
    plt.title("Aligned vs variant router divergence by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "layerwise_js_plots.png", dpi=200)
    plt.close()

    summary_text_lines = []
    if not summary_df.empty:
        grouped = summary_df.groupby("other_perturbation", as_index=False)[["mean_js_yes_answer", "delta_margin"]].mean()
        grouped.to_csv(args.output_dir / "summary_by_perturbation.csv", index=False)
        for _, row in grouped.iterrows():
            summary_text_lines.append(
                f"{row['other_perturbation']}: mean_js_yes_answer={row['mean_js_yes_answer']:.4f}, "
                f"delta_margin={row['delta_margin']:.4f}"
            )

    (args.output_dir / "analysis_summary.txt").write_text("\n".join(summary_text_lines) + "\n")
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
