from __future__ import annotations

import os


def resolve_hf_token(secret_name: str = "HF_TOKEN") -> str | None:
    for env_name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        value = os.getenv(env_name)
        if value:
            return value

    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None

    try:
        value = userdata.get(secret_name)
    except Exception:
        return None
    if value:
        return value
    return None


def configure_hf_token(secret_name: str = "HF_TOKEN") -> str | None:
    token = resolve_hf_token(secret_name=secret_name)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token
