"""Small dataset helpers for configured model-level inference sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptDataset:
    path: Path
    prompts: tuple[str, ...]
    prompt_field: str = "prompt"
    format: str = "jsonl"

    def manifest(self) -> dict[str, Any]:
        lengths = [len(prompt) for prompt in self.prompts]
        return {
            "path": str(self.path),
            "format": self.format,
            "prompt_field": self.prompt_field,
            "num_prompts": len(self.prompts),
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
            "avg_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
        }


def load_prompt_dataset(
    path: str | Path,
    *,
    prompt_field: str = "prompt",
    max_samples: int | None = None,
) -> PromptDataset:
    """Load a local JSONL prompt dataset.

    Each non-empty line must be a JSON object containing ``prompt_field``.  The
    helper is intentionally small and deterministic so profile and verify can
    replay the same prompts without pulling in dataset libraries.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prompt dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Only JSONL prompt datasets are supported for now: {dataset_path}")

    prompts: list[str] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {dataset_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {dataset_path}:{line_no}")
            value = row.get(prompt_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing non-empty field {prompt_field!r} at {dataset_path}:{line_no}")
            prompts.append(value)
            if max_samples is not None and len(prompts) >= max_samples:
                break

    if not prompts:
        raise ValueError(f"No prompts loaded from {dataset_path}")
    return PromptDataset(path=dataset_path, prompts=tuple(prompts), prompt_field=prompt_field)


def load_tokenizer(module_name: str, pretrained: str | None, *, trust_remote_code: bool = True) -> Any | None:
    """Load an AutoTokenizer when available; return None for non-HF-style models."""

    if not pretrained:
        return None
    try:
        module = __import__(module_name, fromlist=["AutoTokenizer"])
    except ImportError:
        return None
    tokenizer_cls = getattr(module, "AutoTokenizer", None)
    if tokenizer_cls is None or not hasattr(tokenizer_cls, "from_pretrained"):
        return None
    kwargs = {"trust_remote_code": trust_remote_code} if module_name == "transformers" else {}
    if module_name == "transformers" and Path(pretrained).expanduser().exists():
        kwargs["local_files_only"] = True
    return tokenizer_cls.from_pretrained(pretrained, **kwargs)


def tokenize_prompts(
    torch: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    *,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    _ensure_padding_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def _ensure_padding_token(tokenizer: Any) -> None:
    """Use the EOS token for padding when decoder-only tokenizers omit one."""

    if getattr(tokenizer, "pad_token", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        return
    try:
        tokenizer.pad_token = eos_token
    except Exception:
        return
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token_id = eos_token_id
        except Exception:
            return
