# Qwen3 AV Router Probe

This project implements a small, Colab-friendly verification experiment for `Qwen/Qwen3-Omni-30B-A3B-Instruct` on an H100.

By default, the project now targets the `Social-IQ-Video` mirror on Hugging Face:

- dataset repo: `PediaMedAI/Social-IQ-Video`
- default split: `validation`
- local layout:
  - `data/social_iq_video/raw_clips`
  - `data/social_iq_video/metadata`
  - `data/social_iq_video/variants`

The task is intentionally tighter than a toy "is this synchronized?" prompt:

> Decide whether the sounds and visuals in a clip come from the same moment of the same event.

That is an audio-visual moment-matching task. It is still small, but it is closer to a benchmark-style verification objective than a pure sync question.

## What This Tests

For each source video, we construct:

- `aligned`: original clip
- `shift_XXXms`: circularly shifted audio remuxed back into the original video
- `gain_+XdB` or `noise_snr_XXdB`: nuisance perturbation that keeps alignment intact

We then prompt the frozen thinker with a binary verification task and teacher-force candidate answers:

- `yes`
- `no`

For each candidate answer, we save:

- candidate log-probability
- candidate margin (`yes_logprob - no_logprob`)
- per-layer router distributions on the answer token(s)
- per-layer router entropies on the answer token(s)

The first question we want to answer is:

1. Does `aligned -> shifted` move router statistics more than `aligned -> nuisance`?

That is the smallest experiment that directly grounds the strongest router-based ideas:

- frozen router-feature verifier
- contrastive sensitivity + invariance

## Files

- `download_social_iq_video.py`: downloads a small `Social-IQ-Video` subset and stores the raw MP4 clips locally
- `prepare_variants.py`: builds shifted and nuisance video variants, then writes `manifest.csv`
- `run_probe.py`: runs Qwen3-Omni thinker with teacher-forced `yes` / `no` answers and saves router features
- `analyze_results.py`: computes layerwise divergences and summary tables/plots
- `experiment_config.py`: prompt and answer configuration
- `requirements-colab.txt`: lightweight Colab requirements

## Recommended Colab Runtime

- GPU: `H100`
- Python: `3.10+`
- Storage: enough for the model cache plus your clips

This experiment is not recommended as a first serious run on a 32 GB local GPU, because quantization/offload can contaminate router-sensitive measurements.

## Quick Start

### 1. Install

```bash
pip install -U pip
pip install -r requirements-colab.txt
```

If the runtime does not already have `ffmpeg`:

```bash
apt-get update && apt-get install -y ffmpeg
```

### 2. Download a Small `Social-IQ-Video` Validation Subset

The default workflow uses the `Social-IQ-Video` validation split and downloads a small set of unique videos.

```bash
python download_social_iq_video.py \
  --split validation \
  --max_videos 12 \
  --selection first
```

This writes:

- `data/social_iq_video/raw_clips/*.mp4`
- `data/social_iq_video/metadata/social_iq_validation_questions.json`
- `data/social_iq_video/metadata/social_iq_validation_questions.csv`

The `Social-IQ-Video` mirror is a copy of the Social-IQ 2.0 challenge data. The Hugging Face dataset page currently shows `987` unique videos and about `8.82k` QA rows across train, validation, and test. Sources:

- [Social-IQ-Video HF mirror](https://huggingface.co/datasets/PediaMedAI/Social-IQ-Video)
- [Social-IQ 2.0 official page](https://cmu-multicomp-lab.github.io/social-iq-2.0/)

### 3. Build Variants

```bash
python prepare_variants.py \
  --shift_ms 800 \
  --nuisance gain \
  --gain_db 3.0
```

This writes:

- `data/social_iq_video/variants/manifest.csv`
- remuxed `shift_800ms.mp4`
- remuxed nuisance variants

### 4. Run the Probe

```bash
python run_probe.py \
  --model_id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --fps 1 \
  --max_pixels $((256 * 28 * 28))
```

### 5. Analyze

```bash
python analyze_results.py
```

## Suggested Narrow Scope For The First Run

If the goal is "finish in about 4 hours and get a real signal", do not start broad.

Use:

- 6 to 12 short clips
- 1 shift value: `800 ms`
- 1 nuisance type: `gain`
- default prompt from `experiment_config.py`
- one run only

That should be enough to see whether router differences are alignment-specific or just generic perturbation effects.

## Expected Outputs

After a successful run you should have:

- `runs/social_iq_router_probe_run1/results.csv`
- one `.npz` feature file per `(clip, variant, answer)`
- `summary_by_clip.csv`
- `layerwise_js_shift_vs_aligned.csv`
- `layerwise_js_plots.png`

## Tightened Task Definition

The prompt is written as moment matching, not generic synchronization:

> Decide whether the sounds and visuals in this clip come from the same moment of the same event. Pay attention to speech, mouth motion, timing, and visible actions. Answer yes or no.

That still targets the alignment phenomenon we care about, but it sounds closer to a realistic verification task.
