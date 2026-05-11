# BIOMIDAS

**Integrated Bioinformatics Platform for Comprehensive Microbiome Analysis**

BIOMIDAS is a web-based analytical platform built with Streamlit that provides end-to-end statistical analysis, visualization, and machine learning for 16S rRNA amplicon sequencing data.

## Features

- **Alpha Diversity** — Shannon, Simpson, Chao1, Faith PD indices
- **Beta Diversity** — PCoA, PERMANOVA (999 permutations), UniFrac
- **Composition Analysis** — CLR normalization, taxonomic bar charts, heatmaps
- **Machine Learning** — Random Forest, XGBoost, LightGBM with SHAP explainability
- **Network Analysis** — Co-occurrence networks, small-world topology (σ)
- **Report Generation** — Automated PDF/HTML reports

## Input Data

BIOMIDAS accepts pre-processed microbiome data:
- OTU/ASV abundance table
- Sample metadata table
- Taxonomic annotation table

> Raw FASTQ processing (QIIME2 / DADA2 pipeline) is not included — use BIOMIDAS after primary bioinformatics processing.

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501)

### Local

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Requirements

See `requirements.txt`. Key dependencies: Streamlit ≥ 1.35, scikit-learn ≥ 1.4, XGBoost, LightGBM, NetworkX, scikit-bio.

## Citation

If you use BIOMIDAS in your research, please cite:

> Аблаев А.Я., Абдурашитов С.Ф.,BIOMIDAS: интегрированная биоинформатическая платформа для комплексного анализа микробиома — мультидатасетная валидация на примере ризосферного микробиома кориандра // Математическая биология и биоинформатика. 2026.

