"""Model registry used by the smoke-test runner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    role: str
    tier: str
    trust_remote_code: bool = False


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("phogpt-4b", "vinai/PhoGPT-4B-Chat", "PhoGPT", "vietnamese-specialized", "3-5B"),
    ModelSpec("qwen35-4b", "Qwen/Qwen3.5-4B", "Qwen3.5", "general-multilingual", "3-5B"),
)


def get_models(keys: list[str] | None = None) -> list[ModelSpec]:
    if not keys:
        return list(MODEL_REGISTRY)
    requested = set(keys)
    available = {model.key: model for model in MODEL_REGISTRY}
    unknown = sorted(requested - available.keys())
    if unknown:
        choices = ", ".join(sorted(available))
        raise ValueError(f"Unknown model key(s): {', '.join(unknown)}. Choose from: {choices}")
    return [model for model in MODEL_REGISTRY if model.key in requested]
