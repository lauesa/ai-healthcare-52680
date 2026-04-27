# Vendor Specialization Claims at 4B: MedGemma vs. Gemma-3 on MIMIC-III MICU Nursing Tasks

**Course:** Graduate Seminar — AI in Healthcare, UT Austin 2026
**Author:** Scott Laue (slaue@utexas.edu)

---

## Project Overview

This project tests a common vendor claim: that models pre-trained on clinical corpora outperform general-purpose models on clinical NLP tasks. I compare **MedGemma-4B-IT** (Google's clinically pre-trained model) against **Gemma-3-4B-IT** (a general instruction-tuned model of identical size and architecture) on three MICU nursing tasks drawn from MIMIC-III, using 637 independent admission records and 13 evaluation metrics.

The result is null-to-negative: the general model matches or outperforms the clinical model on the majority of metrics.

---

## Repository Structure

```
ai-healthcare-52680/
├── src/
│   ├── etl/pipeline.py          # Phase 1: Polars-based ETL → FHIR-lite JSONL
│   ├── inference/engine.py      # Phase 2: vLLM prompt builder + inference runner
│   └── evaluation/metrics.py   # Phase 3: ROUGE-L, Entity F1, Wilcoxon comparison
├── tools/
│   ├── visualise.py             # Generate all 7 paper figures
│   └── sample_jsonl.py          # Reservoir-sample a random record from a JSONL file
├── docs/
│   ├── paper.tex                # ACM sigconf paper
│   ├── references.bib           # Bibliography
│   └── presentation.pptx        # Slide deck with speaker notes
├── data/
│   ├── MIMICIII/                # Raw CSV.gz files (not distributed — requires PhysioNet)
│   └── processed/               # ETL outputs and inference/eval JSONL files (gitignored)
└── .devcontainer/               # Docker dev container (Ubuntu 24.04, Python 3.12)
```

---

## Tasks

| Task | Description | Metrics |
|---|---|---|
| Sub-task 1 | Generate a structured MICU nursing note in AARP format from 12h of EHR context | ROUGE-L, Clinical Entity F1, AARP Section Score |
| Sub-task 2 | Identify abnormal vital signs from most-recent readings | Precision, Recall, F1 |
| Sub-task 3 | Generate a free-form admission summary (hospital course narrative) | ROUGE-L, BERTScore F1, Problem Coverage, Out-of-Range Mention Rate, Medication Mention Rate |

---

## Key Results (n = 637 independent admissions)

| Winner | Metrics |
|---|---|
| Gemma-3-4B-IT | Entity F1 vitals, AARP Score (Sub-task 1); Precision, Recall, F1 (Sub-task 2); Out-of-Range Mention Rate, Medication Mention Rate (Sub-task 3) |
| MedGemma-4B-IT | ROUGE-L, BERTScore F1 (Sub-task 3 only) |
| Draw / n.s. | ROUGE-L and Entity F1 (Sub-task 1, near-floor); Drug Entity F1; Problem Coverage |

---

## Reproducing the Results

> **MIMIC-III requires a credentialed PhysioNet account.** Data cannot be redistributed.
> Sign up at [physionet.org](https://physionet.org) and download MIMIC-III v1.4 CSV files to `data/MIMICIII/`.

### 1. Environment

Open in the provided dev container (VS Code Remote Containers), or manually:

```bash
pip install polars torch vllm spacy rouge-score bert-score scipy pydantic python-pptx
python -m spacy download en_core_sci_sm
```

### 2. ETL

```bash
python -m src.etl.pipeline
```

Outputs `data/processed/fhir_lite_val.jsonl` and split files.

### 3. Inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference.engine \
    --model google/medgemma-4b-it \
    --task notes \
    --split val637opa

CUDA_VISIBLE_DEVICES=0 python -m src.inference.engine \
    --model google/gemma-3-4b-it \
    --task notes \
    --split val637opa
```

Repeat with `--task abnormal` and `--task summary`.

### 4. Evaluation

```bash
python -m src.evaluation.metrics score \
    --inference data/processed/inference_notes_medgemma_val637opa.jsonl \
    --reference data/processed/fhir_lite_val.jsonl
```

### 5. Comparison

```bash
python -m src.evaluation.metrics compare \
    --eval-a data/processed/eval_notes_medgemma_val637opa.jsonl \
    --eval-b data/processed/eval_notes_gemma3_val637opa.jsonl
```

### 6. Figures

```bash
python tools/visualise.py
# outputs figures/academic/ and figures/presentation/
```

---

## Hardware

- **Inference:** NVIDIA RTX 3090 (24 GB VRAM), `CUDA_VISIBLE_DEVICES=0`
- **NER evaluation:** NVIDIA RTX 2080 Ti (11 GB), `CUDA_VISIBLE_DEVICES=1`
- **ETL:** CPU only (Polars lazy API)

---

## Deliverables

- `docs/paper.tex` — ACM sigconf research paper
- `docs/presentation.pptx` — 12-slide deck with full speaker notes
