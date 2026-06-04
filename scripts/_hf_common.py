from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_endpoint() -> str:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    return os.environ["HF_ENDPOINT"]


def sanitize_model_name(model_id: str) -> str:
    name = model_id.strip().replace("\\", "/").rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "model"


def adaptation_cache_dir(model_id: str) -> Path:
    return PROJECT_ROOT / "adaptations" / sanitize_model_name(model_id) / "models"


def default_model_dir(model_id: str) -> Path:
    return PROJECT_ROOT / "model" / sanitize_model_name(model_id)


def local_or_remote_model_ref(model_ref: str) -> str:
    path = Path(model_ref)
    if path.exists() or model_ref.startswith(("/", ".")):
        return str(path.resolve())
    return model_ref
