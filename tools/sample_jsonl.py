"""Print a random line from a JSONL file, pretty-printed as JSON.

Usage:
    python tools/sample_jsonl.py <file.jsonl> [--seed N] [--raw]
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a random line from a JSONL file as pretty-printed JSON."
    )
    parser.add_argument("file", type=Path, help="Path to the .jsonl file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON without pretty-printing.")
    args = parser.parse_args()

    if not args.file.exists():
        parser.error(f"File not found: {args.file}")

    rng = random.Random(args.seed)

    # Reservoir sampling (k=1) — single pass, O(1) memory regardless of file size.
    chosen = None
    with open(args.file, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            if rng.random() < 1.0 / (i + 1):
                chosen = line

    if chosen is None:
        print("File is empty.")
        return

    if args.raw:
        print(chosen)
    else:
        print(json.dumps(json.loads(chosen), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
