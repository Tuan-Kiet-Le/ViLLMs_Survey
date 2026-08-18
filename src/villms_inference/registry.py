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
    status: str = "pending"
    notes: str = ""


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    # Vietnamese-specialized and regional models.
    ModelSpec("phogpt-4b", "vinai/PhoGPT-4B-Chat", "PhoGPT", "vietnamese-specialized", "3-5B", True, notes="Requires remote model code."),
    ModelSpec("vistral-7b", "Viet-Mistral/Vistral-7B-Chat", "Vistral", "vietnamese-specialized", "5-7B"),
    ModelSpec("vinallama-2-7b", "vilm/vinallama-2.7b-chat", "VinaLLaMA", "vietnamese-specialized", "1-3B"),
    ModelSpec("vinallama-7b", "vilm/vinallama-7b-chat", "VinaLLaMA", "vietnamese-specialized", "5-7B"),
    ModelSpec("seallms-v3-1-5b", "SeaLLMs/SeaLLMs-v3-1.5B-Chat", "SeaLLMs v3", "regional-multilingual", "1-3B"),
    ModelSpec("seallms-v3-7b", "SeaLLMs/SeaLLMs-v3-7B-Chat", "SeaLLMs v3", "regional-multilingual", "5-7B"),
    ModelSpec("sailor2-1b", "sail/Sailor2-1B-Chat", "Sailor2", "regional-multilingual", "1-3B"),
    ModelSpec("tiny-aya-water", "CohereLabs/tiny-aya-water", "Tiny Aya", "regional-multilingual", "3-5B"),
    ModelSpec("sea-lion-e2b", "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT", "SEA-LION v4.5", "regional-multilingual", "3-5B"),
    # General multilingual baselines.
    ModelSpec("qwen35-0-8b", "Qwen/Qwen3.5-0.8B", "Qwen3.5", "general-multilingual", "<1B", notes="Verify text-only loading; repository uses a multimodal architecture."),
    ModelSpec("qwen35-2b", "Qwen/Qwen3.5-2B", "Qwen3.5", "general-multilingual", "1-3B", notes="Verify text-only loading; repository uses a multimodal architecture."),
    ModelSpec("qwen35-4b", "Qwen/Qwen3.5-4B", "Qwen3.5", "general-multilingual", "3-5B", notes="Verify text-only loading; repository uses a multimodal architecture."),
    ModelSpec("llama32-1b", "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2", "general-multilingual", "1-3B", notes="May require Hugging Face gated-model access."),
    ModelSpec("llama32-3b", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2", "general-multilingual", "3-5B", notes="May require Hugging Face gated-model access."),
    ModelSpec("phi4-mini", "microsoft/Phi-4-mini-instruct", "Phi-4", "general-multilingual", "3-5B"),
    ModelSpec("gemma-3n-e4b", "google/gemma-3n-E4B-it", "Gemma 3n", "general-multilingual", "3-5B", notes="Verify T4-compatible text-only loading."),
    ModelSpec("falcon-h1-0-5b", "tiiuae/Falcon-H1-0.5B-Instruct", "Falcon-H1", "general-multilingual", "<1B"),
    ModelSpec("falcon-h1-1-5b", "tiiuae/Falcon-H1-1.5B-Instruct", "Falcon-H1", "general-multilingual", "1-3B"),
    ModelSpec("falcon-h1-3b", "tiiuae/Falcon-H1-3B-Instruct", "Falcon-H1", "general-multilingual", "3-5B"),
    ModelSpec("falcon-h1-7b", "tiiuae/Falcon-H1-7B-Instruct", "Falcon-H1", "general-multilingual", "5-7B"),
    ModelSpec("deepseek-r1-distill-1-5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1 Distill", "general-multilingual", "1-3B"),
    ModelSpec("deepseek-r1-distill-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1 Distill", "general-multilingual", "5-7B"),
    ModelSpec("recurrentgemma-2b", "google/recurrentgemma-2b-it", "RecurrentGemma", "general-multilingual", "1-3B"),
    ModelSpec("stablelm-2-1-6b", "stabilityai/stablelm-2-1_6b-chat", "StableLM 2", "general-multilingual", "1-3B"),
    ModelSpec("olmoe-1b-7b", "allenai/OLMoE-1B-7B-0125-Instruct", "OLMoE", "general-multilingual", "1-3B", notes="Mixture-of-experts: 7B total, approximately 1B active."),
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
