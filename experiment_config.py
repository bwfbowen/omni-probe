from __future__ import annotations

DEFAULT_PROMPT = (
    "Decide whether the sounds and visuals in this clip come from the same moment of the same event. "
    "Pay attention to speech, mouth motion, timing, and visible actions. "
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
