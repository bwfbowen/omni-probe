from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiment_config import DEFAULT_ANALYSIS_DIR, DEFAULT_RESULTS_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze router-probe outputs.")
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Probe results CSV. Defaults to the Social-IQ-Video run output path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory for analysis summaries and plots.",
    )
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None, help="Exclusive upper bound.")
    parser.add_argument(
        "--depth_bins",
        type=int,
        default=3,
        help="Number of contiguous depth bins to average over. Defaults to 3 (early/middle/late).",
    )
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


def depth_bin_label(index: int, total_bins: int) -> str:
    if total_bins == 3:
        return ["early", "middle", "late"][index]
    if total_bins == 4:
        return ["early", "mid_early", "mid_late", "late"][index]
    return f"depth_bin_{index + 1}_of_{total_bins}"


def make_depth_bins(num_layers: int, depth_bins: int) -> list[tuple[str, int, int]]:
    if num_layers <= 0:
        return []
    depth_bins = max(1, min(depth_bins, num_layers))
    bin_edges = np.linspace(0, num_layers, num=depth_bins + 1, dtype=int)
    bins: list[tuple[str, int, int]] = []
    for index in range(depth_bins):
        start = int(bin_edges[index])
        end = int(bin_edges[index + 1])
        if end <= start:
            continue
        bins.append((depth_bin_label(index, depth_bins), start, end))
    return bins


def main() -> None:
    args = parse_args()
    if not args.results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV {args.results_csv} does not exist. "
            "Run run_probe.py first or pass --results_csv explicitly."
        )

    import matplotlib.pyplot as plt
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if df.empty:
        raise ValueError("results.csv is empty")

    summary_rows = []
    layerwise_records = []
    depth_summary_rows = []

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
            depth_bins = make_depth_bins(len(layerwise_js), args.depth_bins)
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
            for depth_label, start_layer, end_layer in depth_bins:
                depth_mean_js = float(np.mean(layerwise_js[start_layer:end_layer]))
                depth_summary_rows.append(
                    {
                        "clip_id": clip_id,
                        "other_variant": other_row["variant"],
                        "other_alignment_label": int(other_row["alignment_label"]),
                        "other_perturbation": other_row["perturbation"],
                        "depth_band": depth_label,
                        "layer_start": start_layer,
                        "layer_end_exclusive": end_layer,
                        "mean_js_yes_answer": depth_mean_js,
                        "delta_margin": float(aligned_row["yes_minus_no_margin"] - other_row["yes_minus_no_margin"]),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    layerwise_df = pd.DataFrame(layerwise_records)
    depth_summary_df = pd.DataFrame(depth_summary_rows)
    summary_df.to_csv(args.output_dir / "summary_by_clip.csv", index=False)
    layerwise_df.to_csv(args.output_dir / "layerwise_js_shift_vs_aligned.csv", index=False)
    depth_summary_df.to_csv(args.output_dir / "summary_by_clip_and_depth.csv", index=False)

    aggregate = (
        layerwise_df.groupby(["other_perturbation", "layer_index"], as_index=False)["js_yes_answer"]
        .mean()
        .sort_values(["other_perturbation", "layer_index"])
    )
    aggregate.to_csv(args.output_dir / "layerwise_js_aggregate.csv", index=False)

    depth_aggregate = (
        depth_summary_df.groupby(
            ["other_perturbation", "depth_band", "layer_start", "layer_end_exclusive"], as_index=False
        )[["mean_js_yes_answer", "delta_margin"]]
        .mean()
        .sort_values(["other_perturbation", "layer_start"])
    )
    depth_aggregate.to_csv(args.output_dir / "summary_by_perturbation_and_depth.csv", index=False)

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

    if not depth_aggregate.empty:
        plt.figure(figsize=(10, 5))
        perturbations = list(depth_aggregate["other_perturbation"].unique())
        depth_labels = list(dict.fromkeys(depth_aggregate["depth_band"]))
        x = np.arange(len(depth_labels))
        width = 0.8 / max(len(perturbations), 1)
        for idx, perturbation in enumerate(perturbations):
            part = depth_aggregate.loc[depth_aggregate["other_perturbation"] == perturbation]
            part = part.set_index("depth_band").reindex(depth_labels).reset_index()
            offset = (idx - (len(perturbations) - 1) / 2.0) * width
            plt.bar(x + offset, part["mean_js_yes_answer"], width=width, label=perturbation)
        plt.xticks(x, depth_labels)
        plt.ylabel("Mean JS divergence on `yes` answer router distribution")
        plt.title("Aligned vs variant router divergence by depth band")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "depth_band_js_plots.png", dpi=200)
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
    if not depth_aggregate.empty:
        summary_text_lines.append("")
        summary_text_lines.append("Depth-band means:")
        for perturbation, part in depth_aggregate.groupby("other_perturbation"):
            depth_text = ", ".join(
                f"{row['depth_band']}={row['mean_js_yes_answer']:.4f}" for _, row in part.iterrows()
            )
            summary_text_lines.append(f"{perturbation}: {depth_text}")

    (args.output_dir / "analysis_summary.txt").write_text("\n".join(summary_text_lines) + "\n")
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
