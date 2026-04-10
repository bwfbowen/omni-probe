from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiment_config import DEFAULT_RAW_CLIPS_DIR, DEFAULT_VARIANTS_DIR

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shifted and nuisance AV variants.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_RAW_CLIPS_DIR,
        help="Directory containing source video clips. Defaults to the Social-IQ-Video raw clip folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_VARIANTS_DIR,
        help="Directory for generated variants. Defaults to the Social-IQ-Video variant folder.",
    )
    parser.add_argument(
        "--shift_ms",
        type=int,
        nargs="+",
        default=[800],
        help="One or more circular audio shifts in milliseconds.",
    )
    parser.add_argument(
        "--nuisance",
        choices=["gain", "noise", "none"],
        default="none",
        help="Nuisance perturbation that should preserve alignment.",
    )
    parser.add_argument(
        "--mismatch",
        choices=["swap", "permute"],
        nargs="*",
        default=["swap", "permute"],
        help="Extra strong mismatch variants to create for each clip.",
    )
    parser.add_argument("--gain_db", type=float, default=3.0, help="Gain change used when nuisance=gain.")
    parser.add_argument("--noise_snr_db", type=float, default=28.0, help="SNR used when nuisance=noise.")
    parser.add_argument(
        "--permute_chunk_ms",
        type=float,
        default=1000.0,
        help="Chunk size in milliseconds used when mismatch includes permute.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Temporary wav extraction sample rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise nuisance.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_audio(src_video: Path, dst_wav: Path, sample_rate: int) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(dst_wav),
        ]
    )


def remux_video_with_audio(src_video: Path, src_wav: Path, dst_video: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_video),
            "-i",
            str(src_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(dst_video),
        ]
    )


def circular_shift_audio(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0:
        return np.array(audio, copy=True)
    return np.roll(audio, shift_samples, axis=0)


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    multiplier = float(10 ** (gain_db / 20.0))
    return np.clip(audio * multiplier, -1.0, 1.0)


def apply_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(audio))) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(audio.dtype)
    return np.clip(audio + noise, -1.0, 1.0)


def match_audio_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if audio.shape[0] == target_len:
        return np.array(audio, copy=True)
    if audio.shape[0] > target_len:
        return np.array(audio[:target_len], copy=True)

    repeats = int(math.ceil(target_len / max(audio.shape[0], 1)))
    tiled = np.tile(audio, repeats)
    return np.array(tiled[:target_len], copy=True)


def permute_audio_chunks(audio: np.ndarray, chunk_samples: int, seed: int) -> np.ndarray:
    if chunk_samples <= 0 or audio.shape[0] <= 1:
        return np.array(audio, copy=True)

    chunks = [audio[start : start + chunk_samples] for start in range(0, audio.shape[0], chunk_samples)]
    if len(chunks) < 2:
        return np.array(audio, copy=True)

    rng = np.random.default_rng(seed)
    order = np.arange(len(chunks))
    for _ in range(8):
        rng.shuffle(order)
        if not np.array_equal(order, np.arange(len(chunks))):
            break
    else:
        order = np.roll(order, 1)

    return np.concatenate([chunks[index] for index in order], axis=0)[: audio.shape[0]].astype(np.float32, copy=False)


def iter_videos(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)


def main() -> None:
    args = parse_args()
    import soundfile as sf

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory {args.input_dir} does not exist. "
            "Run download_social_iq_video.py first or pass --input_dir explicitly."
        )
    videos = iter_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {args.input_dir}")

    rows: list[dict[str, str | int | float]] = []
    mismatch_set = set(args.mismatch)

    with tempfile.TemporaryDirectory(prefix="prepare_variants_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        import soundfile as sf

        audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        for src_video in videos:
            wav_path = tmpdir / f"{src_video.stem}.wav"
            extract_audio(src_video, wav_path, args.sample_rate)
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio_cache[src_video.stem] = (np.asarray(audio, dtype=np.float32), int(sample_rate))

        for clip_index, src_video in enumerate(videos):
            clip_id = src_video.stem
            clip_dir = args.output_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            audio, sample_rate = audio_cache[clip_id]

            rows.append(
                {
                    "clip_id": clip_id,
                    "variant": "aligned",
                    "video_path": str(src_video.resolve()),
                    "source_video_path": str(src_video.resolve()),
                    "audio_source_clip_id": clip_id,
                    "alignment_label": 1,
                    "perturbation": "none",
                    "shift_ms": 0,
                }
            )

            for shift_ms in args.shift_ms:
                shift_samples = int(round(sample_rate * (shift_ms / 1000.0)))
                shifted_audio = circular_shift_audio(audio, shift_samples)
                shifted_wav = tmpdir / f"{clip_id}_shift_{shift_ms}ms.wav"
                shifted_video = clip_dir / f"{clip_id}_shift_{shift_ms}ms.mp4"
                sf.write(shifted_wav, shifted_audio, sample_rate)
                remux_video_with_audio(src_video, shifted_wav, shifted_video)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "variant": f"shift_{shift_ms}ms",
                        "video_path": str(shifted_video.resolve()),
                        "source_video_path": str(src_video.resolve()),
                        "audio_source_clip_id": clip_id,
                        "alignment_label": 0,
                        "perturbation": "shift",
                        "shift_ms": shift_ms,
                    }
                )

            if "swap" in mismatch_set and len(videos) > 1:
                donor_video = videos[(clip_index + 1) % len(videos)]
                donor_clip_id = donor_video.stem
                donor_audio, _ = audio_cache[donor_clip_id]
                swapped_audio = match_audio_length(donor_audio, audio.shape[0])
                swapped_wav = tmpdir / f"{clip_id}_swap_from_{donor_clip_id}.wav"
                swapped_video = clip_dir / f"{clip_id}_swap_from_{donor_clip_id}.mp4"
                sf.write(swapped_wav, swapped_audio, sample_rate)
                remux_video_with_audio(src_video, swapped_wav, swapped_video)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "variant": f"swap_from_{donor_clip_id}",
                        "video_path": str(swapped_video.resolve()),
                        "source_video_path": str(src_video.resolve()),
                        "audio_source_clip_id": donor_clip_id,
                        "alignment_label": 0,
                        "perturbation": "swap",
                        "shift_ms": 0,
                    }
                )

            if "permute" in mismatch_set:
                chunk_samples = int(round(sample_rate * (args.permute_chunk_ms / 1000.0)))
                permuted_audio = permute_audio_chunks(audio, chunk_samples=chunk_samples, seed=args.seed + clip_index)
                permute_name = f"permute_{int(args.permute_chunk_ms)}ms"
                permuted_wav = tmpdir / f"{clip_id}_{permute_name}.wav"
                permuted_video = clip_dir / f"{clip_id}_{permute_name}.mp4"
                sf.write(permuted_wav, permuted_audio, sample_rate)
                remux_video_with_audio(src_video, permuted_wav, permuted_video)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "variant": permute_name,
                        "video_path": str(permuted_video.resolve()),
                        "source_video_path": str(src_video.resolve()),
                        "audio_source_clip_id": clip_id,
                        "alignment_label": 0,
                        "perturbation": "permute",
                        "shift_ms": 0,
                    }
                )

            if args.nuisance == "gain":
                nuisance_audio = apply_gain(audio, args.gain_db)
                nuisance_name = f"gain_{args.gain_db:+.1f}dB".replace(".", "p")
            elif args.nuisance == "noise":
                nuisance_audio = apply_noise(audio, args.noise_snr_db, seed=args.seed + clip_index)
                nuisance_name = f"noise_snr_{args.noise_snr_db:.1f}dB".replace(".", "p")
            else:
                nuisance_audio = None
                nuisance_name = ""

            if nuisance_audio is not None:
                nuisance_wav = tmpdir / f"{clip_id}_{nuisance_name}.wav"
                nuisance_video = clip_dir / f"{clip_id}_{nuisance_name}.mp4"
                sf.write(nuisance_wav, nuisance_audio, sample_rate)
                remux_video_with_audio(src_video, nuisance_wav, nuisance_video)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "variant": nuisance_name,
                        "video_path": str(nuisance_video.resolve()),
                        "source_video_path": str(src_video.resolve()),
                        "audio_source_clip_id": clip_id,
                        "alignment_label": 1,
                        "perturbation": args.nuisance,
                        "shift_ms": 0,
                    }
                )

    fieldnames = [
        "clip_id",
        "variant",
        "video_path",
        "source_video_path",
        "audio_source_clip_id",
        "alignment_label",
        "perturbation",
        "shift_ms",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Prepared {len(rows)} rows across {len(videos)} source clips")


if __name__ == "__main__":
    main()
