"""Command-line interface for the ViLLMs smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_prompts, write_jsonl
from .registry import MODEL_REGISTRY, get_models
from .runner import run_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ViLLMs inference smoke test")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--models", help="Comma-separated registry keys; defaults to all")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/smoke.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/smoke/results.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_models:
        for model in MODEL_REGISTRY:
            print(f"{model.key}\t{model.model_id}\t{model.role}\t{model.tier}")
        return
    prompts = load_prompts(args.prompts)
    keys = [key.strip() for key in args.models.split(",")] if args.models else None
    records = []
    for model in get_models(keys):
        print(f"Running {model.key} ({model.model_id})", flush=True)
        try:
            records.extend(run_model(model, prompts, max_new_tokens=args.max_new_tokens, seed=args.seed))
        except Exception as exc:
            records.append({"model_key": model.key, "model_id": model.model_id, "status": "error", "error_type": type(exc).__name__, "error": str(exc)})
            print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} record(s) to {args.output}")
