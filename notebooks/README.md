# Notebooks

This directory contains Jupyter notebooks that demonstrate how to run the data synthesis workflows.

## Core notebook

- `qa_synthesis.ipynb` — minimal QA synthesis walkthrough using the Python API.

## Example runs

The `examples/` folder mirrors the experiments in the repository root. Use these as references for configuring
production runs with different domains, prompts, and providers.

See `examples/WORKFLOWS.md` for workflow patterns that explain how to adapt any example notebook for
**source-backed** or **no-source** generation.

| Notebook | Focus | Suggested config |
| --- | --- |
| `examples/corpusgen-contract-gemini-prodv1.ipynb` | Contract corpus generation with Gemini. | `configs/corpusgen_contract_gemini.yaml` |
| `examples/corpusgen-legal-gemini-prodv1.ipynb` | Legal corpus generation with Gemini. | `configs/corpusgen_legal.yaml` |
| `examples/corpusqa-syndeepthink-gemini-prodv1.ipynb` | QA generation with deep-thinking prompts. | `configs/corpusqa_deepthinking.yaml` |
| `examples/gemini-any-syn-prod-v1.ipynb` | General synthesis run with Gemini. | `configs/gemini_any_syn.yaml` |
| `examples/gemini-any-syndeepthink-prod-v1.ipynb` | General synthesis run using deep-thinking prompts. | `configs/gemini_any_syndeepthink.yaml` |
| `examples/gemini-any-synthink-prod-v1.ipynb` | General synthesis run using structured reasoning prompts. | `configs/gemini_any_synthink.yaml` |
| `examples/legal-qa-syn-gemini-prod-v1.ipynb` | Legal QA synthesis with Gemini. | `configs/legal_qa_gemini.yaml` |
| `examples/legallqagen-gemini-prodv1.ipynb` | Legal question generation with Gemini. | `configs/legallqagen_gemini.yaml` |

## Tips

- Start with `qa_synthesis.ipynb` to validate your environment setup.
- Copy a notebook from `examples/` when you need a production run template.
- If you move or rename notebooks, update this index to keep it in sync.
