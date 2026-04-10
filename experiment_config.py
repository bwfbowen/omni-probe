from __future__ import annotations

from pathlib import Path


SOCIAL_IQ_DATASET_REPO_ID = "PediaMedAI/Social-IQ-Video"
SOCIAL_IQ_DEFAULT_SPLIT = "validation"

DEFAULT_DATA_ROOT = Path("data/social_iq_video")
DEFAULT_RAW_CLIPS_DIR = DEFAULT_DATA_ROOT / "raw_clips"
DEFAULT_METADATA_DIR = DEFAULT_DATA_ROOT / "metadata"
DEFAULT_VARIANTS_DIR = DEFAULT_DATA_ROOT / "variants"

DEFAULT_RESULTS_DIR = Path("runs/social_iq_router_probe_run1")
DEFAULT_RESULTS_CSV = DEFAULT_RESULTS_DIR / "results.csv"
DEFAULT_ANALYSIS_DIR = DEFAULT_RESULTS_DIR / "analysis"

DEFAULT_PROMPT = (
    "Decide whether the sounds and visuals in this social interaction clip come from the same moment of the same event. "
    "Pay attention to speech, turn-taking, mouth motion, reactions, and timing. "
    "Answer yes or no."
)

DEFAULT_ANSWER_TEXT = {
    "yes": " yes",
    "no": " no",
}


def make_conversation(video_path: str, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
