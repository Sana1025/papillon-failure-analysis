# PAPILLON Failure Analysis

Research project: **Characterizing When PAPILLON Fails and Developing Mitigation Methods for Direct PII Leakage**

## Overview

[PAPILLON](https://arxiv.org/abs/2410.17127) is a privacy-preserving LLM delegation framework that uses a local LLM to sanitize user queries before sending them to a more powerful remote LLM. This project characterizes when PAPILLON fails to remove direct PII from sanitized prompts, analyzes the root causes of failure, and evaluates mitigation strategies.

Evaluated on the [PUPA benchmark](https://huggingface.co/datasets/Columbia-NLP/PUPA) (901 real user queries with PII annotations).

## Research Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | 🔄 In progress | Establish PAPILLON baseline |
| 2 | 🔄 In progress | Characterize and categorize failure cases |
| 3 | ⏳ Planned | Analyze causes of failure |
| 4 | ⏳ Planned | Develop mitigation approach |
| 5 | ⏳ Planned | Experimental evaluation |

## Repository Structure

```
papillon-failure-analysis/
├── leakage_auditor.py     # Deterministic PII leakage auditor (Phase 2)
├── test_auditor.py        # Validation suite for the auditor
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
# Clone PAPILLON baseline (use the reproducibility branch)
git clone --branch papillon_v1.0 https://github.com/siyan-sylvia-li/PAPILLON.git

# Create and activate a virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key (needed for PAPILLON's remote LLM calls)
export OPENAI_API_KEY="your-key-here"
```

## Leakage Auditor

`leakage_auditor.py` is a deterministic, LLM-free auditor that checks each annotated PII unit against a sanitized prompt using a cascade of matching strategies:

| Level | Failure Type | What it catches |
|-------|-------------|-----------------|
| 1 | `exact_copy` | PII found verbatim |
| 2 | `case_variant` | Found after case normalization |
| 3 | `punct_variant` | Found after stripping punctuation |
| 4 | `space_variant` | Found after collapsing whitespace |
| 5 | `combo_variant` | Found after combining all normalizations |
| 6 | `partial_identifier` | ≥75% of a multi-token PII present as substring |
| — | `uncertain` | PII unit ≤ 4 chars (too short to assess reliably) |

### Usage

```bash
# Run against PAPILLON pipeline output
python leakage_auditor.py \
    --input  papillon_output.csv \
    --san_col sanitized_prompt \
    --output  audit_query.csv \
    --pii_output audit_pii.csv \
    --summary audit_summary.json

# Validate the auditor itself
python test_auditor.py
```

### Input CSV format

| Column | Description |
|--------|-------------|
| `user_query` | Original user prompt (with PII) |
| `pii_units` | Pipe-delimited (`\|\|`) PII annotations |
| `sanitized_prompt` | The prompt actually sent to the remote LLM |
| `predicted_category` | Task category (optional, for breakdown stats) |
| `conversation_hash` | Row identifier (optional) |

## Key Findings So Far

- PUPA-TNB has **15 rows with missing `redacted_query`** and annotation artefacts (malformed PII strings, numbered prefixes)
- The `metric_finegrained` function in `run_dspy_optimization_llama.py` contains a **typo** (`ppi_str` instead of `pii_str`) that silently passes `None` to the leakage judge — the paper's fine-grained leakage scores may be unreliable
- **7 PII units in `redacted_query`** were not redacted by the PUPA annotators themselves, indicating the benchmark's own privacy coverage has gaps
- The `evaluate_papillon.py` script has a **list-length misalignment bug** that can corrupt the output CSV

## Citation

```bibtex
@inproceedings{li-etal-2025-papillon,
    title = "{PAPILLON}: {P}riv{A}cy {P}reservation from {I}nternet-based and {L}oca{L} Language m{O}del e{N}sembles",
    author = "Li, Siyan Sylvia and Raghuram, Vethavikashini Chithrra and Celikyilmaz, Asli and Xu, Jieyu and He, Xiang Ren and Peng, Nanyun and Zhou, Fei",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2025",
}
```
