"""Input and output helpers for reproducible inference runs."""

import json
from pathlib import Path
from typing import Any


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
            raise ValueError(f"Line {line_number} must contain non-empty 'id' and 'prompt' fields")
        prompts.append({"id": str(item["id"]), "prompt": str(item["prompt"])})
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
