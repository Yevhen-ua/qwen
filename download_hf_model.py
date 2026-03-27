#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a model repository from Hugging Face to a local folder."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face repo id, for example stabilityai/stable-diffusion-3.5-medium",
    )
    parser.add_argument(
        "--dest",
        default="/home/total/ai/sd35/models/stable-diffusion-3.5-medium",
        help="Destination folder",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, branch, tag, or commit",
    )
    parser.add_argument(
        "--cache-dir",
        default="/home/total/ai/hf-cache",
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional allow patterns, e.g. '*.json' '*.safetensors'",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional ignore patterns",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF token. By default uses HF_TOKEN environment variable or the token saved by hf auth login.",
    )

    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo:      {args.repo_id}")
    print(f"To:        {dest}")
    print(f"Cache:     {cache_dir}")
    print(f"Revision:  {args.revision or 'main'}")
    if args.include:
        print(f"Include:   {args.include}")
    if args.exclude:
        print(f"Exclude:   {args.exclude}")
    print()

    try:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(dest),
            cache_dir=str(cache_dir),
            revision=args.revision,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
            token=args.token
        )
    except Exception as exc:
        print("Download failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "\nFor gated models, first accept the model license in your browser and then run 'hf auth login' or export HF_TOKEN=...",
            file=sys.stderr,
        )
        return 1

    print("Done.")
    print(f"Local path: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())