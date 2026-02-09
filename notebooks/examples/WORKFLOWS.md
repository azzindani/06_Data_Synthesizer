# Notebook workflow patterns (source vs. no-source)

This document mirrors the example notebooks by describing the two primary workflows used across the
experiments: **source-backed** generation and **no-source** generation. Use these patterns to adapt any
example notebook to your own domain and data.

## 1) No-source workflow (prompt-only)

Use this when you want the model to generate content from prompts alone.

**Common use cases**
- Bootstrapping a new domain with no existing corpus.
- Rapid topic exploration (cross-topic or cross-relation prompts).

**Typical steps**
1. Define topic lists or prompt templates in your config or notebook.
2. Select synthesis mode (`qa`, `deep_thinking`, or `corpus`).
3. Generate and validate outputs.

## 2) Source-backed workflow (grounded)

Use this when you have an existing corpus or QA pairs and want to synthesize or augment data grounded
in those sources.

**Common use cases**
- Document-to-QA generation (ground-truth anchored QA).
- QA augmentation from existing pairs.

**Typical steps**
1. Load source documents or QA pairs in the notebook.
2. Create prompts that include source context (citations or excerpts).
3. Generate and validate outputs.

## How to apply these patterns to the example notebooks

Pick any notebook under `notebooks/examples/` and:
- **No-source**: remove any source loading step and rely on prompt templates plus topic lists.
- **Source-backed**: add a data-loading cell and pass the source content into the prompt template.

If you add new notebooks, include a short note at the top describing whether it is **no-source** or
**source-backed** so the catalog stays aligned.
