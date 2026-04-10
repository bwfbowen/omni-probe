from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

from experiment_config import DEFAULT_ANSWER_TEXT, DEFAULT_PROMPT, make_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3 AV router probe.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--save_full_router_probs", action="store_true")
    parser.add_argument("--prompt_text", type=str, default=DEFAULT_PROMPT)
    return parser.parse_args()


def choose_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch_dtype: {name}")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def clone_with_appended_answer(inputs: dict[str, Any], answer_ids: torch.Tensor) -> dict[str, Any]:
    appended = dict(inputs)
    answer_ids = answer_ids.unsqueeze(0)
    appended["input_ids"] = torch.cat([inputs["input_ids"], answer_ids], dim=1)
    if "attention_mask" in inputs:
        answer_mask = torch.ones(
            (inputs["attention_mask"].shape[0], answer_ids.shape[1]),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        appended["attention_mask"] = torch.cat([inputs["attention_mask"], answer_mask], dim=1)
    return appended


def normalize_router_tuple(
    router_logits: tuple[torch.Tensor, ...],
    batch_size: int,
    sequence_length: int,
) -> list[torch.Tensor]:
    normalized = []
    for layer_gate in router_logits:
        if layer_gate.dim() == 2:
            normalized.append(layer_gate.view(batch_size, sequence_length, -1))
        elif layer_gate.dim() == 3:
            normalized.append(layer_gate)
        else:
            raise ValueError(f"Unexpected router tensor shape: {tuple(layer_gate.shape)}")
    return normalized


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def compute_candidate_logprob(
    logits: torch.Tensor,
    appended_input_ids: torch.Tensor,
    prompt_len: int,
    answer_len: int,
) -> float:
    log_probs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for offset in range(answer_len):
        predict_position = prompt_len - 1 + offset
        target_position = prompt_len + offset
        token_id = appended_input_ids[0, target_position]
        total += float(log_probs[predict_position, token_id].item())
    return total


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def build_processor(model_id: str, min_pixels: int | None, max_pixels: int | None) -> AutoProcessor:
    kwargs = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def extract_answer_features(
    outputs: Any,
    prompt_len: int,
    answer_len: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    if outputs.router_logits is None:
        raise RuntimeError(
            "router_logits is missing from model outputs. "
            "Make sure the installed Transformers version supports output capture for Qwen3-Omni."
        )

    answer_positions = list(range(prompt_len, prompt_len + answer_len))
    normalized_layers = normalize_router_tuple(outputs.router_logits, batch_size=batch_size, sequence_length=seq_len)

    layer_prob_means = []
    layer_entropy_means = []
    layer_top1 = []

    for layer_probs in normalized_layers:
        answer_probs = layer_probs[0, answer_positions, :]
        mean_probs = answer_probs.mean(dim=0)
        layer_prob_means.append(mean_probs.detach().cpu().numpy().astype(np.float32))
        layer_entropy_means.append(float(entropy_from_probs(answer_probs).mean().item()))
        layer_top1.append(int(mean_probs.argmax().item()))

    return {
        "router_probs_mean": np.stack(layer_prob_means, axis=0),
        "router_entropy_mean": np.asarray(layer_entropy_means, dtype=np.float32),
        "router_top1": np.asarray(layer_top1, dtype=np.int32),
        "answer_positions": np.asarray(answer_positions, dtype=np.int32),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest_csv)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    processor = build_processor(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=choose_dtype(args.torch_dtype),
        device_map=args.device_map,
    )
    model.eval()
    device = next(model.parameters()).device

    tokenizer = processor.tokenizer
    answer_id_map = {
        name: tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        for name, text in DEFAULT_ANSWER_TEXT.items()
    }

    all_results: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in rows:
            clip_id = row["clip_id"]
            variant = row["variant"]
            video_path = row["video_path"]

            conversation = make_conversation(video_path, args.prompt_text)
            inputs = processor.apply_chat_template(
                conversation,
                load_audio_from_video=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                fps=args.fps,
                padding=True,
                use_audio_in_video=True,
            )
            inputs = move_batch_to_device(inputs, device)
            prompt_len = int(inputs["input_ids"].shape[1])

            candidate_metrics: dict[str, dict[str, Any]] = {}
            for answer_name, answer_ids in answer_id_map.items():
                appended_inputs = clone_with_appended_answer(inputs, answer_ids)
                outputs = model(
                    **appended_inputs,
                    use_audio_in_video=True,
                    output_router_logits=True,
                    return_dict=True,
                )

                candidate_logprob = compute_candidate_logprob(
                    logits=outputs.logits,
                    appended_input_ids=appended_inputs["input_ids"],
                    prompt_len=prompt_len,
                    answer_len=int(answer_ids.numel()),
                )
                feature_payload = extract_answer_features(
                    outputs=outputs,
                    prompt_len=prompt_len,
                    answer_len=int(answer_ids.numel()),
                    batch_size=int(appended_inputs["input_ids"].shape[0]),
                    seq_len=int(appended_inputs["input_ids"].shape[1]),
                )
                feature_payload["candidate_logprob"] = np.asarray([candidate_logprob], dtype=np.float32)
                feature_payload["answer_token_ids"] = answer_ids.detach().cpu().numpy().astype(np.int32)
                if not args.save_full_router_probs:
                    feature_payload.pop("router_top1", None)

                feature_name = f"{safe_name(clip_id)}__{safe_name(variant)}__{answer_name}.npz"
                feature_path = feature_dir / feature_name
                np.savez_compressed(feature_path, **feature_payload)

                candidate_metrics[answer_name] = {
                    "logprob": candidate_logprob,
                    "feature_path": str(feature_path.resolve()),
                    "answer_token_count": int(answer_ids.numel()),
                }

            yes_logprob = candidate_metrics["yes"]["logprob"]
            no_logprob = candidate_metrics["no"]["logprob"]
            margin = yes_logprob - no_logprob
            prediction = "yes" if margin >= 0 else "no"

            result_row = {
                **row,
                "prompt_text": args.prompt_text,
                "prompt_len": prompt_len,
                "yes_logprob": yes_logprob,
                "no_logprob": no_logprob,
                "yes_minus_no_margin": margin,
                "prediction": prediction,
                "yes_feature_path": candidate_metrics["yes"]["feature_path"],
                "no_feature_path": candidate_metrics["no"]["feature_path"],
                "yes_answer_token_count": candidate_metrics["yes"]["answer_token_count"],
                "no_answer_token_count": candidate_metrics["no"]["answer_token_count"],
            }
            all_results.append(result_row)
            print(json.dumps({"clip_id": clip_id, "variant": variant, "prediction": prediction, "margin": margin}))

    results_csv = args.output_dir / "results.csv"
    fieldnames = list(all_results[0].keys()) if all_results else []
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Wrote {results_csv}")


if __name__ == "__main__":
    main()
