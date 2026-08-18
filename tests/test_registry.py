from villms_inference.registry import get_models


def test_default_registry_contains_smoke_pair() -> None:
    assert [model.key for model in get_models()] == ["phogpt-4b", "qwen35-4b"]


def test_registry_filter() -> None:
    models = get_models(["qwen35-4b"])
    assert models[0].model_id == "Qwen/Qwen3.5-4B"
