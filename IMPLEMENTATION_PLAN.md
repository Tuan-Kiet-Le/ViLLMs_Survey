# ViLLMs Inference Implementation Plan

This document is the running record of our discussion and decisions about
building an inference setup for the models listed in
`docs/references/ViLLMs.md`. We will update it as we resolve questions.

## Current context

- Repository: `ViLLMs_Survey`
- The repository includes the user-facing README, the source model reference
  table in `docs/references/ViLLMs.md`, and a runnable Transformers harness.
- The runtime registry in `src/villms_inference/registry.py` covers three roles:
  Vietnamese-specialized, Southeast-Asian/regional multilingual, and general
  multilingual baselines. It records model IDs, families, roles, size tiers,
  and model-specific loading or compatibility notes.
- The registry's standard checkpoints range from approximately 0.5B to 7B
  parameters; the source reference table remains the research inventory.
- The goal is to support repeatable inference across multiple models, with
  comparable inputs, generation settings, saved outputs, and measurements.

## Initial design direction

The implementation should be model-config driven rather than one-off scripts.
Each model should have a registry entry containing its Hugging Face identifier,
family, language/region category, parameter-size bucket, expected model type,
conversation format, and any model-specific loading requirements.

The first version should prioritize a reliable single-model command and then
add batch execution over the registry. It should make it possible to run a
small smoke test before downloading or evaluating the complete model set.

The inference record should preserve, at minimum:

- model identifier and exact revision;
- prompt/input identifier and text;
- tokenizer and chat-template path used;
- generation parameters and runtime/device information;
- generated text;
- latency and token counts where available;
- warnings, errors, and load failures.

Model downloads, caches, quantization, and device placement must be explicit
so that results can be reproduced on another machine. Gated or restricted
models must be detected and reported rather than silently skipped.

## Research direction for a strong paper

The current project should not be framed as only “running a list of models.”
That would be useful engineering, but it is unlikely to be a sufficient Q1 or
A* contribution by itself. The scientific focus should instead be Vietnamese
Small Language Models (SLMs), with multilingual models as comparison points.
The dual-T4 Kaggle environment is the controlled experimental platform, not
the motivation or definition of the research problem.

Cleaner working framings include:

- “A Systematic Evaluation of Vietnamese Small Language Models”;
- “Vietnamese Small Language Models: Quality, Robustness, and Efficiency
  Across Model Families”;
- “Evaluating Vietnamese-Capable Small Language Models Under a Unified
  Protocol.”

The final title should be chosen only after we define the SLM size boundary and
the paper’s specific gap. Quality, robustness, and efficiency should answer a
Vietnamese SLM research question rather than make the hardware sound like the
contribution.

Potential research questions:

1. How do Vietnamese-specialized SLMs compare with multilingual SLM baselines
   at similar parameter counts under identical prompts and decoding
   conditions?
2. Which capabilities are preserved or lost as Vietnamese models become
   smaller: knowledge, comprehension, reasoning, instruction following, and
   conversation?
3. How sensitive are rankings to prompt language, few-shot demonstrations,
   chat-template handling, decoding settings, quantization, and evaluator
   choice?
4. Which Vietnamese SLMs provide the best quality/latency/memory trade-off for
   realistic deployment scenarios?

The paper will need a clear novelty claim beyond a leaderboard. Possible
contributions include a carefully controlled cross-family comparison, a
Vietnamese-specific robustness and prompt-sensitivity protocol, an
efficiency-quality Pareto analysis for SLM deployment, or a new small
evaluation set that fills a documented gap in existing Vietnamese benchmarks.
We should not claim a new benchmark unless we establish that existing suites
do not cover the intended research question.

### Recommended benchmark layers

Use layers rather than one overloaded score:

- Vietnamese capability: VMLU, which covers general knowledge, reading
  comprehension, reasoning, and conversational skills.
- Vietnamese task continuity/comparability: ViLLM-Eval or relevant VLSP task
  subsets, with exact task versions and prompts recorded.
- General capability controls: a small, justified subset of established tasks
  such as MMLU, GSM8K, and IFEval, chosen to avoid turning the study into an
  unbounded benchmark sweep.
- Efficiency: load time, peak VRAM, throughput, time to first token, output
  length, and failures under a fixed hardware/software configuration.
- Robustness: prompt-language variants, deterministic repeated runs, and
  sensitivity to chat-template and decoding settings.

VMLU is especially relevant because it is a recent ACL 2025 benchmark toolkit
for Vietnamese LLMs. ViLLM-Eval is useful for continuity with the Vietnamese
LLM evaluation/shared-task line. Both must be checked for licensing, available
evaluation code, versioning, train/test overlap, and whether their scoring
protocols are appropriate for the selected model types.

### Experimental discipline needed for publication

- Pre-register or freeze the model list, revisions, prompts, metrics, and
  decoding settings before the final comparison.
- Report exact model revisions, tokenizer/chat-template behavior, software
  versions, GPU assignment, dtype, quantization, and batch settings.
- Separate zero-shot, few-shot, and instruction-following results; do not
  compare them as one number.
- Report confidence intervals or bootstrap intervals where the benchmark
  supports them, not only average scores.
- Include an error taxonomy and representative Vietnamese examples, not only
  aggregate tables.
- Report failed loads, invalid generations, truncation, and answer-extraction
  failures explicitly.
- Treat evaluator-model scores as secondary evidence and validate them against
  human ratings or objective metrics where possible.
- Compare models within sensible size/family groups before making global
  claims.

### Initial implementation strategy on Kaggle T4x2

The first milestone should use only one or two representative models and a
very small, fixed sample. Its purpose is to verify the complete path:

1. environment setup;
2. deterministic model/tokenizer loading;
3. correct chat-template application;
4. one benchmark adapter;
5. generation and answer extraction;
6. metric computation;
7. raw-output and metadata persistence;
8. repeatable result summarization.

Only after this smoke benchmark is correct should we optimize throughput. On
two T4 GPUs, we should test whether the chosen backend actually supports the
desired multi-GPU strategy; simply having two GPUs does not guarantee that a
Transformers process will use both. We should benchmark single-GPU loading,
data parallel evaluation, and model sharding separately, while keeping a
single canonical correctness configuration.

The optimization target should be a measured Pareto trade-off among score,
runtime, peak VRAM, and cost—not “maximum speed” alone. Quantized runs should
be reported as separate conditions because quantization may change quality.

## Decisions still needed

### 1. What is the first inference objective?

Possible scopes include:

- interactive chat/CLI inference;
- a fixed prompt suite for qualitative comparison;
- benchmark evaluation against an existing dataset;
- Vietnamese ASR-adjacent or speech-related testing;
- all of the above in stages.

### 2. Which models are in the first wave?

The reference table currently lists about two dozen model entries. Running all
of them will require substantially different memory, download time, and
possibly model-specific code. We should choose a small representative subset
for the first end-to-end path, then expand the registry.

### 3. What hardware and runtime are available?

Please specify the target environment, especially GPU model and VRAM, system
RAM, operating system, CUDA availability, and whether CPU-only inference is
required. This determines whether we use standard Transformers, quantized
weights, vLLM, llama.cpp, or a layered fallback.

### 4. What quality and performance signals matter?

Candidate measurements are Vietnamese instruction following, factuality,
translation, reasoning, response length, time to first token, tokens per
second, peak memory, and total load/inference time. We should decide which are
required and which are optional diagnostics.

### 5. What input and output contract do we want?

We need to choose the prompt format (single-turn, multi-turn chat, or both),
the input format (JSONL, YAML, plain text, or code-defined cases), and the
output format (JSONL, Markdown reports, SQLite, or a combination).

### 6. How should model-specific behavior be handled?

Some listed models may have custom chat templates, remote code requirements,
multimodal capabilities, reasoning modes, or incompatible architectures. We
need a policy for whether to support these immediately, mark them as pending,
or exclude them from comparable scoring while still allowing smoke tests.

### 7. What does “perfectly” mean for this project?

For this plan, I interpret it as: correct loading, correct tokenizer/template
use, safe device and dtype selection, reproducible generation, clear failure
reporting, and comparable logging. If you mean production serving, maximum
throughput, or research-grade benchmarking, the architecture and validation
requirements will be different.

## Proposed discussion order

1. Confirm the first objective and representative first-wave models.
2. Confirm hardware/runtime constraints.
3. Define the test prompts or benchmark dataset.
4. Define metrics and the result schema.
5. Implement the smallest end-to-end smoke test.
6. Add model adapters and registry entries incrementally.
7. Add batch runs, comparison reports, and reproducibility checks.

## Discussion log

### 2026-08-18 — Initial setup

- User wants to test multiple models listed in `docs/references/ViLLMs.md`.
- User requested that the discussion be recorded in this file before
  implementation begins.
- Initial repository inspection found no existing inference code or test
  harness.
- Open questions are recorded above for the next discussion turn.

### 2026-08-18 — User constraints and research direction

- Target compute is Kaggle with two NVIDIA T4 GPUs. The exact setup code will
  be provided later by the user.
- The primary objective is benchmark evaluation, not interactive chat.
- The initial implementation should run only one or two models on a small
  sample to validate the complete pipeline before scaling out.
- The user wants the system optimized as much as possible, subject to keeping
  evaluation correct, reproducible, and comparable.
- The plan now recommends separating correctness validation from throughput
  optimization and treating quality/latency/memory/cost as a Pareto analysis.
- Initial benchmark candidates identified for investigation are VMLU and
  ViLLM-Eval, supplemented by a small justified set of general capability
  tasks and explicit efficiency/robustness measurements.

### 2026-08-18 — SLM framing clarification

- The user clarified that the scientific focus should be Vietnamese SLMs;
  “constrained dual-T4 inference” is an awkward research framing.
- Dual-T4 is the experimental platform and should be reported as a
  reproducibility detail, while SLMs remain the central research subject.
- The working direction is now a systematic evaluation of Vietnamese SLMs,
  with multilingual models as matched comparison points.
- The study scope is models below 7B parameters, split into size tiers.
- Vietnamese-specialized and multilingual models should be compared at matched
  parameter sizes while still covering multiple model families and their
  latest eligible releases.
- The paper will target general Vietnamese capability judgments rather than a
  single deployment domain. The benchmark suite will be specified later by
  the user.
- A new Vietnamese test subset is out of scope for this paper; the study will
  use existing benchmarks with a rigorous, reproducible protocol.
- The approximately 7B tier is included because the target dual-T4 platform
  can run these models; the scope is therefore up to and including 7B rather
  than strictly below 7B.
- The agreed size tiers are `<1B`, `1–<3B`, `3–<5B`, and `5–7B`.
- “Latest” means the latest eligible model available at the time of the
  experiment. The exact experiment date and model revisions must be recorded
  in the final paper and result metadata.

### Questions for the next discussion turn

1. When several checkpoints from one family fit a tier, should we select one
   representative model per family or include multiple variants?

Here, a representative or “flagship” checkpoint means one selected model
release from a family, such as one Qwen, one SeaLLMs, one VinaLLaMA, or one
Llama checkpoint. Including multiple variants from the same family gives more
coverage but can make the comparison dominated by a few large families. A
balanced default is one representative checkpoint per family per size tier,
with additional variants included only when they answer a specific question.

### 2026-08-18 — Size and release policy decisions

- The approximately 7B tier is included because it can run on the target
  dual-T4 platform.
- Agreed size tiers: `<1B`, `1–<3B`, `3–<5B`, and `5–7B`.
- “Latest” means the latest eligible release available at experiment time;
  exact revisions and the experiment date must be frozen in the metadata.
- The meaning of “representative/flagship checkpoint per family” was clarified:
  it means selecting one model release from each family in a tier, rather than
  evaluating every same-family variant by default.
- The user selected the broader policy: evaluate all eligible models in each
  size tier, including multiple checkpoints from the same family when they are
  in scope. Family-level balancing can be handled during analysis rather than
  by excluding models from the raw evaluation.

### Questions for the next discussion turn

1. Should the benchmark include only instruction/chat-tuned checkpoints, or
   also base/pretrained checkpoints? For general capability judgments, the
   recommended default is instruction/chat-tuned models only, because base
   models are not directly comparable in conversational benchmark settings.

### 2026-08-18 — Checkpoint type decision

- The user agreed to evaluate instruction/chat-tuned checkpoints only.
- Base/pretrained checkpoints are excluded from the primary comparison because
  the study targets general judgments and conversational benchmark behavior.

### Questions for the next discussion turn

1. Should the primary protocol be zero-shot only, or should it include
   zero-shot plus few-shot and prompt-robustness conditions?

### 2026-08-18 — Practical evaluation protocol decision

- The user does not consider few-shot prompting or dedicated prompt-robustness
  testing representative of real-world use for this project.
- The primary protocol will therefore be zero-shot, single-turn evaluation
  using realistic user prompts.
- Few-shot and prompt-robustness conditions are out of scope for the main
  paper unless the user later identifies a concrete practical use case for
  them.

### Questions for the next discussion turn

1. Should the primary task format be single-turn question answering only, or
   should it also include realistic multi-turn conversations?

### 2026-08-18 — Smoke-test scope decision

- The first implementation milestone will use exactly one single-turn sample
  per model, only to validate the end-to-end inference and evaluation path.
- Multi-turn conversations and broader task coverage are deferred until the
  user provides the benchmark details.
- The first sample is a pipeline smoke test, not a meaningful scientific
  result and must not be presented as the paper’s evaluation.

### Model-selection policy

The model list should have two layers:

1. **Study universe:** all eligible models that satisfy the inclusion criteria.
2. **Smoke-test subset:** a small, deliberately selected set used to validate
   the inference pipeline before scaling to the full universe.

Eligibility criteria for the study universe:

- instruction-tuned or chat-tuned checkpoint;
- open weights that can be downloaded and run reproducibly;
- text-only causal language model for the main study;
- Vietnamese capability or a justified multilingual-baseline role;
- actual size within one of the agreed tiers, with the 7B boundary documented;
- usable tokenizer/chat template and a supported inference path;
- identifiable model revision, license, and release metadata.

Model families should be tagged into three comparison roles:

- Vietnamese-specialized;
- Southeast-Asian or regional multilingual;
- general multilingual baseline.

The final study can include all eligible models, including multiple checkpoints
from one family. Analysis should report both per-model results and summaries
within size tier and comparison role, so a family with many checkpoints does
not automatically dominate the conclusions.

The initial smoke-test subset should use selected registry models in comparable
size tiers and across comparison roles, subject to confirming their exact
checkpoint identifiers, licenses, chat templates, and availability at the
experiment date.

The model table should not be expanded merely because a model is new. New
models should be added when they fill a family, language, or size-tier gap and
can be evaluated under the same protocol. Deprecated checkpoints should be
retained only when historical comparability is a stated goal.

## Inference optimization strategy

Optimization should be separated into two explicitly named modes:

### Canonical comparison mode

This mode exists to make model scores comparable. It should use one shared
loading and generation policy wherever the architectures support it:

- the same Transformers-based inference path initially;
- the same input prompt and chat-message schema;
- each model's official tokenizer and chat template;
- fixed dtype and generation settings appropriate for T4 hardware;
- deterministic decoding for the smoke test;
- no quantization in the first quality comparison unless every model can be
  handled equivalently;
- no concurrent requests when measuring per-model latency;
- complete capture of model revision, software versions, device placement,
  token counts, latency, memory, and errors.

The canonical mode is the source of quality results. It should remain stable
even if later optimizations are added.

### Optimized deployment mode

This mode measures how efficiently each model can run in practice. It may test:

- FP16 and quantized variants where supported;
- attention and KV-cache settings;
- batch size and padding strategy;
- model compilation or optimized attention kernels;
- vLLM or another serving backend when the model architecture and chat
  template are supported;
- one-model-per-GPU parallel execution for independent requests;
- model sharding across both GPUs for models that need it;
- model-loading and tokenizer-cache reuse.

Every optimization must be recorded as a condition, not silently mixed into the
canonical results. A quality regression caused by quantization or a backend
change must remain visible.

### Recommended optimization order

1. Make one model produce the correct answer with the official template.
2. Add structured metadata and raw-output persistence.
3. Measure baseline load time, generation latency, peak VRAM, and token rate.
4. Add the second matched-size model and verify that both use the same harness.
5. Optimize memory with safe dtype/device placement.
6. Test quantization as a separate condition.
7. Test batching or parallel multi-GPU execution only when there are enough
   requests for it to matter.
8. Test an optimized serving backend and compare both quality and speed with
   the canonical path.

For the first one-prompt-per-model smoke test, batching is not expected to
provide a meaningful throughput benefit. The priority is fast model swapping,
low peak memory, correct template handling, and reliable failure recovery.

### Multi-GPU policy

The two T4 GPUs can be used in different ways, which must not be conflated:

- **Sequential isolation:** one model at a time on one GPU; simplest and best
  for clean per-model latency measurements.
- **Parallel model evaluation:** independent models or requests assigned to
  separate GPUs; useful for reducing wall-clock experiment time, but it changes
  contention and should not define the canonical latency metric.
- **Model sharding:** one model split across both GPUs; useful when one GPU
  cannot hold the model, but it introduces communication overhead and should be
  reported separately.

The first smoke test should use sequential isolation unless the later Kaggle
setup reveals a strong reason to do otherwise.

### 2026-08-18 — Model-selection discussion

- The user confirmed that one prompt will be run on each selected model for
  the first smoke test.
- The user is concerned about model selection; the plan now separates the
  full eligible study universe from the small initial smoke-test subset.
- The smoke-test candidates are selected from the standard-checkpoint registry
  by role and size tier.
- Current model discovery indicates that model cards can distinguish newer and
  deprecated checkpoints within a family, so release status and exact
  revisions must be recorded before final inclusion. For example, the
  SeaLLMs model card identifies newer v2/v2.5/v3 variants and marks an older
  7B checkpoint as deprecated.

### 2026-08-18 — Optimization discussion started

- The user wants to discuss optimization of each model's run before expanding
  the implementation.
- The plan now separates canonical comparison mode from optimized deployment
  mode, so efficiency work cannot silently invalidate quality comparisons.
- The initial one-prompt-per-model smoke test prioritizes correctness, memory,
  template handling, metadata, and failure recovery; batching is deferred until
  there are enough requests for throughput optimization to matter.

### Quantization explanation

Quantization stores model weights and sometimes activations using fewer bits.
Instead of representing values mainly in FP16, a quantized run may use INT8 or
INT4 representations. This can substantially reduce VRAM usage and sometimes
improve speed, allowing larger models to fit on limited hardware.

The trade-off is that lower numerical precision can change the generated
answer, reduce quality, or introduce backend-specific behavior. Quantization
methods are also not identical: GPTQ, AWQ, bitsandbytes, and other methods can
have different memory, speed, and quality characteristics.

Therefore, the first canonical comparison will use a common non-quantized
configuration where possible. Quantized configurations (including GGUF and
AWQ variants) are deferred until the standard-checkpoint workflow is stable;
they will be measured later as separate optimized conditions and reported with
their own model format, quantization method, dtype, backend, quality score,
VRAM, and latency.

### 2026-08-18 — Basic harness implemented

- Added a `uv`-managed Python project with Transformers, Accelerate, PyTorch,
  and pytest dependencies.
- Added a standard-checkpoint registry covering Vietnamese-specialized,
  Southeast-Asian/regional, and general multilingual models across the agreed
  size tiers, with loading and compatibility notes.
- Added a JSONL prompt input with one Vietnamese single-turn sample.
- Added a sequential canonical runner using the official tokenizer/chat
  template, greedy decoding, no quantization, and FP16 on CUDA when available.
- Added JSONL result persistence with response, token counts, load/generation
  timing, device, dtype, software versions, CUDA version, and peak VRAM.
- Added model failure capture so one failed model does not discard other
  results.
- Local validation completed: `uv sync`, 3 tests passing, and CLI registry
  listing verified. Real model downloads and inference remain for Kaggle.

### Recommended next step

Run the first Kaggle preflight and one-prompt smoke test across the registry.
The user should provide the Kaggle setup code/notebook, after which the next
work should be:

1. Check Python, CUDA, GPU visibility, disk space, and available VRAM.
2. Install or adapt the locked dependencies without changing the canonical
   inference settings.
3. Confirm access to the selected standard-checkpoint repositories.
4. Run exactly one Vietnamese prompt per selected registry model.
5. Inspect raw responses, chat-template behavior, load failures, latency,
   output-token counts, and peak VRAM.
6. Fix correctness or compatibility issues before adding benchmark adapters.
7. Only after the standard-checkpoint smoke test passes, evaluate quantized
   variants or add benchmark adapters.

The smoke run is a pipeline validation, not a research result. It should not
be used to draw quality conclusions from a single prompt.

### 2026-08-18 — Kaggle preflight added

- Added `scripts/kaggle_preflight.py` to inspect Python packages, CUDA,
  visible GPUs, VRAM, `nvidia-smi`, disk space, and optional model-config
  accessibility before downloading full model weights.
- The next Kaggle command should be:
  `uv run python scripts/kaggle_preflight.py --check-models`.

### Questions for the next discussion turn

1. What parameter boundary should define an SLM in this study: at most 1B,
   at most 3B, at most 7B, or a tiered definition?
2. Should the primary comparison be Vietnamese-specialized versus multilingual
   models at matched size, or a broader ranking of all models in the table?
3. Which deployment scenario motivates the study most: local consumer devices,
   edge/server inference, education, public services, or general Vietnamese
   assistants?
4. Do you want the paper to contribute a new Vietnamese evaluation subset, or
   focus first on a rigorous analysis using existing benchmarks?
