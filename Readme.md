# ATGBUILDER: Feature-Assisted Graph Learning for Activity Transition Graph Construction with Seed Supervision

This repository contains the artifact for **ATGBUILDER**, a feature-assisted graph-learning framework for **seed-supervised Activity Transition Graph (ATG) construction**.

It includes:
- Code for **LLM-based UI summary generation** (activity/layout and widget-related summaries)
- Code for **embedding generation**, **feature fusion**, and **ATG link prediction**
- Scripts/configs for **data preprocessing** and **experiments**
- Supplementary details referenced in the main paper ([`Supplementary Material for ATGBuilder.pdf`](./Supplementary%20Material%20for%20ATGBuilder.pdf)), including:
  - (1) the full LLM prompt template for summary generation; 
  - (2) additional details on the GCN vs. GIN comparison; 
  - (3) additional results on negative sampling strategies.

> **Note (Anonymized artifact):** This repository is prepared for anonymized review. Any private tokens/keys are removed and must be configured by users.

---

## Repository Structure

This project contains two main modules:
```text
  ├── summarygeneration/        # LLM-based summary generation
  └── atgbuilder/               # Embedding + prediction + experiments
```

### 1) `summarygeneration/` — Summary Generation

This module generates compact, functionality-oriented summaries for activities/widgets.

Key files:
- `com/blind/aise/dataprocessing/summarized/generators/ATGGeneratorTemplate.java`  
  Template code of the summary generator.
- `src/main/resources/application.properties`  
  Configuration of the LLM API (e.g., endpoint/key). **You must replace placeholders with your own settings.**
- `src/main/resources/templates/prompt_template.yaml`  
  Prompt templates used for summary generation. You may modify them for different summarization styles.

**Outputs**
- Generated summaries will be written under the configured output directory (see configs).  
  These summaries are later consumed by the `atgbuilder/` pipeline for embedding and prediction.

---

### 2) `atgbuilder/` — Embedding, Prediction, and Experiments

This module implements the ATG construction pipeline (feature fusion + graph learning + link prediction).

#### 2.1 Seed ATGs and Summaries
- The repository includes example/processed seed ATGs and generated summaries under:
  - `atgbuilder/verify_seed_atgs/`  *(or the actual directory name in your repo)*
    - Seed ATG files (raw or processed)
    - Generated summaries used by the pipeline

> If your artifact stores summaries/ATGs elsewhere, please update this path to a **relative path inside the repository**.

#### 2.2 `atgbuilder/embedding/` — Embedding Pipeline
This directory contains code and configs for generating embeddings for:
- activity identifiers (names/types)
- LLM-generated summaries
- widget attributes (if available)

Key configs:
- `atgbuilder/embedding/config/embedding_config.yaml`
- `atgbuilder/embedding/config/pipeline_config.yaml`  
  These define **input/output paths** and pipeline options (e.g., which embeddings to generate).

Core script:
- `atgbuilder/embedding/pipeline/run_pipeline.py`  
  Main entry for embedding generation. Running it produces embedding files that are used by downstream prediction.

#### 2.3 `atgbuilder/prediction/` — Transition Prediction
This directory contains the link-prediction models and training/evaluation pipeline for ATG construction.

#### 2.4 `atgbuilder/experiments/` — Experiment Drivers
This directory contains experiment scripts (RQ-oriented runners), result collection, and plotting utilities.

---

## Data Availability

Due to space constraints, the full **Frontmatter** dataset is not included in this repository.  
Please download it from the official source:

- Frontmatter dataset: https://zenodo.org/records/5084655
- ATG benchmark: please refer to the original paper for details: Activity Transition Graph Generation: How Far Are We?