from pathlib import Path

from villms_inference.io import load_prompts


def test_load_prompts(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"p1","prompt":"Xin chào"}\n', encoding="utf-8")
    assert load_prompts(path) == [{"id": "p1", "prompt": "Xin chào"}]
