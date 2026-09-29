#!/usr/bin/env python3
"""Validate MLX-LM chat JSONL files before training."""
import json
import sys
from pathlib import Path


def validate(path: Path) -> int:
    if not path.exists():
        print(f"Missing: {path}", file=sys.stderr)
        return 1
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                return 1
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                print(f"{path}:{line_no}: expected a messages list with user and assistant", file=sys.stderr)
                return 1
            roles = [m.get("role") for m in messages if isinstance(m, dict)]
            if len(roles) != len(messages) or "user" not in roles or roles[-1] != "assistant":
                print(f"{path}:{line_no}: messages must include user and end with assistant", file=sys.stderr)
                return 1
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                print(f"{path}:{line_no}: every message needs non-empty text content", file=sys.stderr)
                return 1
            count += 1
    if count == 0:
        print(f"{path}: no examples", file=sys.stderr)
        return 1
    print(f"{path}: {count} valid examples")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "data"
    paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else [root / "train.jsonl", root / "valid.jsonl"]
    return max(validate(path) for path in paths)


if __name__ == "__main__":
    raise SystemExit(main())
