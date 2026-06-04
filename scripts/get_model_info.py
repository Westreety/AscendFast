#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from _hf_common import adaptation_cache_dir, configure_hf_endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Hugging Face model metadata and config.json."
    )
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--source", choices=["huggingface"], default="huggingface")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--files-metadata", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def sibling_names(info: Any) -> list[str]:
    siblings = getattr(info, "siblings", None) or []
    names: list[str] = []
    for item in siblings:
        names.append(getattr(item, "rfilename", str(item)))
    return names


def main() -> int:
    args = parse_args()
    endpoint = configure_hf_endpoint()
    cache_dir = Path(args.cache_dir) if args.cache_dir else adaptation_cache_dir(args.model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(endpoint=endpoint)
    info = api.model_info(
        args.model_id,
        revision=args.revision,
        files_metadata=args.files_metadata,
    )

    config: dict[str, Any] | None = None
    config_error: str | None = None
    try:
        config_path = hf_hub_download(
            args.model_id,
            "config.json",
            revision=args.revision,
            cache_dir=cache_dir,
            endpoint=endpoint,
        )
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        config_error = str(exc)

    result = {
        "model_id": getattr(info, "modelId", args.model_id),
        "sha": getattr(info, "sha", None),
        "author": getattr(info, "author", None),
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": getattr(info, "tags", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "last_modified": serialize_datetime(getattr(info, "lastModified", None)),
        "siblings": sibling_names(info),
        "cache_dir": str(cache_dir),
        "hf_endpoint": endpoint,
        "config": config,
        "config_error": config_error,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"model_id: {result['model_id']}")
    print(f"sha: {result['sha']}")
    print(f"library_name: {result['library_name']}")
    print(f"pipeline_tag: {result['pipeline_tag']}")
    print(f"downloads: {result['downloads']}")
    print(f"likes: {result['likes']}")
    print(f"last_modified: {result['last_modified']}")
    print(f"cache_dir: {result['cache_dir']}")
    print(f"hf_endpoint: {result['hf_endpoint']}")
    if config is not None:
        print("config:")
        print(json.dumps(config, ensure_ascii=False, indent=2))
    else:
        print(f"config_error: {config_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
