from villms_inference.registry import get_models


def test_default_registry_contains_smoke_pair() -> None:
    keys = [model.key for model in get_models()]
    assert keys[:2] == ["phogpt-4b", "vistral-7b"]
    assert "qwen35-4b" in keys
    assert len(keys) >= 20


def test_registry_filter() -> None:
    models = get_models(["qwen35-4b"])
    assert models[0].model_id == "Qwen/Qwen3.5-4B"
