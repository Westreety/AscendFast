#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from _hf_common import configure_hf_endpoint, default_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot into this project.")
    parser.add_argument("model_id", help="Hugging Face model id, for example Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", default=None, help="Default: model/<repo-name>")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = configure_hf_endpoint()
    local_dir = Path(args.local_dir) if args.local_dir else default_model_dir(args.model_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=local_dir,
        endpoint=endpoint,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.max_workers,
        token=args.token,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    print(f"downloaded_to: {path}")
    print(f"hf_endpoint: {endpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
