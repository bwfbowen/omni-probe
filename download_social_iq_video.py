from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from experiment_config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_METADATA_DIR,
    DEFAULT_RAW_CLIPS_DIR,
    SOCIAL_IQ_DATASET_REPO_ID,
    SOCIAL_IQ_DEFAULT_SPLIT,
)


QA_FILENAME_BY_SPLIT = {
    "train": "qa/qa_train.json",
    "validation": "qa/qa_val.json",
    "test": "qa/qa_test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small Social-IQ-Video subset for the router probe.")
    parser.add_argument("--repo_id", type=str, default=SOCIAL_IQ_DATASET_REPO_ID)
    parser.add_argument("--split", choices=sorted(QA_FILENAME_BY_SPLIT), default=SOCIAL_IQ_DEFAULT_SPLIT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max_videos", type=int, default=12, help="Number of unique validation videos to download.")
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="How to choose unique videos from the split.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed when selection=random.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload and overwrite existing local MP4 files.")
    return parser.parse_args()


def load_qa_rows(path: Path) -> list[dict]:
    text = path.read_text()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse {path} as JSON or JSONL. "
                    f"First invalid JSONL record is on line {line_number}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object per line in {path}, got {type(row).__name__} on line {line_number}")
            rows.append(row)
        if rows:
            return rows
        raise ValueError(f"{path} is empty or not valid JSON/JSONL")

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["data", "questions", "qa", "rows"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported QA JSON structure in {path}")


def ordered_unique_video_ids(rows: list[dict]) -> list[str]:
    seen = set()
    ordered_ids = []
    for row in rows:
        vid_name = row.get("vid_name")
        if not vid_name or vid_name in seen:
            continue
        seen.add(vid_name)
        ordered_ids.append(vid_name)
    return ordered_ids


def available_video_ids(repo_id: str) -> set[str]:
    from huggingface_hub import list_repo_files

    repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    available_ids = set()
    for file_path in repo_files:
        if not file_path.startswith("video/"):
            continue
        if not file_path.endswith(".mp4"):
            continue
        available_ids.add(Path(file_path).stem)
    return available_ids


def choose_video_ids(all_ids: list[str], max_videos: int, selection: str, seed: int) -> list[str]:
    if max_videos <= 0:
        raise ValueError("--max_videos must be positive")
    if max_videos >= len(all_ids):
        return all_ids
    if selection == "first":
        return all_ids[:max_videos]
    rng = random.Random(seed)
    chosen = list(all_ids)
    rng.shuffle(chosen)
    return sorted(chosen[:max_videos])


def write_selected_metadata(rows: list[dict], selected_ids: set[str], metadata_dir: Path, split: str) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in rows if row.get("vid_name") in selected_ids]
    json_path = metadata_dir / f"social_iq_{split}_questions.json"
    json_path.write_text(json.dumps(selected_rows, indent=2))

    csv_path = metadata_dir / f"social_iq_{split}_questions.csv"
    if selected_rows:
        fieldnames = sorted({key for row in selected_rows for key in row})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_rows)

    videos_txt = metadata_dir / f"social_iq_{split}_video_ids.txt"
    videos_txt.write_text("\n".join(sorted(selected_ids)) + "\n")


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_clips_dir = args.output_dir / DEFAULT_RAW_CLIPS_DIR.name
    metadata_dir = args.output_dir / DEFAULT_METADATA_DIR.name
    raw_clips_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    qa_filename = QA_FILENAME_BY_SPLIT[args.split]
    qa_cache_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=qa_filename)
    qa_rows = load_qa_rows(Path(qa_cache_path))
    qa_video_ids = ordered_unique_video_ids(qa_rows)
    repo_video_ids = available_video_ids(args.repo_id)
    all_video_ids = [video_id for video_id in qa_video_ids if video_id in repo_video_ids]
    missing_video_ids = sorted(set(qa_video_ids) - repo_video_ids)

    if not all_video_ids:
        raise ValueError(
            f"No downloadable video IDs were found for split={args.split} in repo {args.repo_id}. "
            "The QA metadata and video tree appear to be out of sync."
        )

    selected_video_ids = choose_video_ids(
        all_ids=all_video_ids,
        max_videos=args.max_videos,
        selection=args.selection,
        seed=args.seed,
    )

    for vid_name in selected_video_ids:
        cached_video = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=f"video/{vid_name}.mp4")
        dst_video = raw_clips_dir / f"{vid_name}.mp4"
        if dst_video.exists() and not args.overwrite:
            continue
        shutil.copy2(cached_video, dst_video)

    write_selected_metadata(
        rows=qa_rows,
        selected_ids=set(selected_video_ids),
        metadata_dir=metadata_dir,
        split=args.split,
    )
    if missing_video_ids:
        missing_path = metadata_dir / f"social_iq_{args.split}_missing_video_ids.txt"
        missing_path.write_text("\n".join(missing_video_ids) + "\n")
        print(
            f"Skipped {len(missing_video_ids)} QA video IDs with no matching video/*.mp4 entry in the dataset repo. "
            f"Wrote the list to {missing_path}"
        )

    print(f"Downloaded {len(selected_video_ids)} Social-IQ-Video clips to {raw_clips_dir}")
    print(f"Saved QA metadata to {metadata_dir}")


if __name__ == "__main__":
    main()
