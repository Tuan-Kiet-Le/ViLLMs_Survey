"""Single-process Transformers inference runner."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from typing import Any

from .registry import ModelSpec


def run_model(model_spec: ModelSpec, prompts: list[dict[str, str]], *, max_new_tokens: int = 128, seed: int = 0) -> list[dict[str, Any]]:
    """Load one model and run all prompts sequentially in canonical mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id, trust_remote_code=model_spec.trust_remote_code)
    processor = None
    model_loader = AutoModelForCausalLM
    if model_spec.loader == "image_text_to_text":
        processor = AutoProcessor.from_pretrained(model_spec.model_id, trust_remote_code=model_spec.trust_remote_code)
        model_loader = AutoModelForImageTextToText
    model = model_loader.from_pretrained(
        model_spec.model_id,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
        trust_remote_code=model_spec.trust_remote_code,
    )
    if not use_cuda:
        model.to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt["prompt"]}]
        if processor is not None:
            encoded = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        elif getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            encoded = tokenizer(prompt["prompt"], return_tensors="pt").input_ids
        encoded = encoded.to(input_device)
        if use_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(encoded, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, encoded.shape[-1] :]
        token_count = int(new_tokens.shape[-1])
        records.append({
            "model_key": model_spec.key,
            "model_id": model_spec.model_id,
            "family": model_spec.family,
            "role": model_spec.role,
            "tier": model_spec.tier,
            "prompt_id": prompt["id"],
            "prompt": prompt["prompt"],
            "response": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "input_tokens": int(encoded.shape[-1]),
            "output_tokens": token_count,
            "generation_seconds": elapsed,
            "output_tokens_per_second": token_count / elapsed if elapsed else None,
            "load_seconds": load_seconds,
            "device": str(input_device),
            "dtype": str(dtype).replace("torch.", ""),
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
        })
    return records
