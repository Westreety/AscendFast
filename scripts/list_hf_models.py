#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huggingface_hub import HfApi

from _hf_common import configure_hf_endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Hugging Face models.")
    parser.add_argument("--search", default=None, help="Text search, for example Qwen2.5.")
    parser.add_argument("--author", default=None)
    parser.add_argument("--task", default=None, help="Pipeline tag, for example text-generation.")
    parser.add_argument("--library", default=None, help="Library name, for example transformers.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sort", default="downloads")
    parser.add_argument("--desc", action="store_true", help="Sort descending.")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> int:
    args = parse_args()
    endpoint = configure_hf_endpoint()
    api = HfApi(endpoint=endpoint)
    models = api.list_models(
        search=args.search,
        author=args.author,
        task=args.task,
        library=args.library,
        limit=args.limit,
        sort=args.sort,
        direction=-1 if args.desc else None,
        full=args.full,
    )

    print(f"HF_ENDPOINT={endpoint}")
    print("model_id\tpipeline_tag\tlibrary\tdownloads\tlikes\tlast_modified")
    for model in models:
        print(
            "\t".join(
                [
                    fmt(getattr(model, "modelId", None)),
                    fmt(getattr(model, "pipeline_tag", None)),
                    fmt(getattr(model, "library_name", None)),
                    fmt(getattr(model, "downloads", None)),
                    fmt(getattr(model, "likes", None)),
                    fmt(getattr(model, "lastModified", None)),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
