from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shifted and nuisance AV variants.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing source video clips.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for generated variants.")
    parser.add_argument(
        "--shift_ms",
        type=int,
        nargs="+",
        default=[800],
        help="One or more circular audio shifts in milliseconds.",
    )
    parser.add_argument(
        "--nuisance",
        choices=["gain", "noise"],
        default="gain",
        help="Nuisance perturbation that should preserve alignment.",
    )
    parser.add_argument("--gain_db", type=float, default=3.0, help="Gain change used when nuisance=gain.")
    parser.add_argument("--noise_snr_db", type=float, default=28.0, help="SNR used when nuisance=noise.")
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


def iter_videos(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)


def main() -> None:
    args = parse_args()
    import soundfile as sf

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    videos = iter_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {args.input_dir}")

    rows: list[dict[str, str | int | float]] = []

    for clip_index, src_video in enumerate(videos):
        clip_id = src_video.stem
        clip_dir = args.output_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        rows.append(
            {
                "clip_id": clip_id,
                "variant": "aligned",
                "video_path": str(src_video.resolve()),
                "source_video_path": str(src_video.resolve()),
                "alignment_label": 1,
                "perturbation": "none",
                "shift_ms": 0,
            }
        )

        with tempfile.TemporaryDirectory(prefix=f"{clip_id}_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            wav_path = tmpdir / f"{clip_id}.wav"
            extract_audio(src_video, wav_path, args.sample_rate)
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

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
                        "alignment_label": 0,
                        "perturbation": "shift",
                        "shift_ms": shift_ms,
                    }
                )

            if args.nuisance == "gain":
                nuisance_audio = apply_gain(audio, args.gain_db)
                nuisance_name = f"gain_{args.gain_db:+.1f}dB".replace(".", "p")
            else:
                nuisance_audio = apply_noise(audio, args.noise_snr_db, seed=args.seed + clip_index)
                nuisance_name = f"noise_snr_{args.noise_snr_db:.1f}dB".replace(".", "p")

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
