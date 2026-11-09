# Advanced Multimodal and Multilingual News Recommendation Framework
### *Robust, Explainable, and Personalized Content Delivery*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

---

## 1. Project Overview

The **Advanced Multimodal and Multilingual News Recommendation Framework** is an intelligent, scalable, and interpretable recommendation system designed to provide personalized news delivery across diverse languages and modalities.

### Key Innovations:
- **Multilingual Semantic Representations**: Semantic alignment across **English**, **Hindi**, and **Assamese** using multilingual transformer models (XLM-RoBERTa / Multilingual Sentence-Transformers).
- **Multimodal Feature Fusion**: Joint modeling of news content by extracting dense textual embeddings and deep visual representations (ResNet50 / ViT) through cross-attention and weighted gating.
- **Hybrid Recommendation Pipeline**: Combines Content-Based filtering, Collaborative Filtering, and Temporal User-Interest decay.
- **Cross-Lingual Recommendation**: Recommends semantically related articles across different languages based on user latent interests.
- **Explainable AI (XAI)**: Generates human-readable, transparent justifications for each recommended news item.
- **Fairness & Diversity-Aware Ranking**: Mitigates popularity bias, filter bubbles, and echo chambers via Maximal Marginal Relevance (MMR) intra-list diversity.
- **Recommendation Stability & Robustness**: Robust under sparse interaction and cold-start conditions.

---

## 2. Project Directory Structure

```text
d:\Dhanjit sir project\
│
├── config/
│   ├── __init__.py
│   └── config.py               # Central configuration, hyperparameters & paths
│
├── data/
│   ├── raw/                    # Raw news data (MIND, regional datasets)
│   ├── processed/              # Cleaned, tokenized & normalized news articles
│   ├── embeddings/             # Cached dense embeddings (text & image vectors)
│   └── sample/                 # Multilingual sample dataset for testing
│
├── database/
│   ├── __init__.py
│   ├── schema.sql              # SQLite schema for Users, Articles, History, Logs
│   └── db_manager.py           # SQLite manager & query operations
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/          # Phase 2: Multilingual cleaning & dataset loader
│   ├── models/                 # Phase 3 & 4: Text (XLM-R) and Image (ResNet) encoders
│   ├── recommendation/         # Phase 5, 6, 7, 8: Content, Collaborative, Hybrid & Cross-lingual
│   ├── explainability/         # Phase 9: XAI explanation module
│   ├── evaluation/             # Phase 10: Ranking, diversity, stability metrics
│   └── utils/                  # Helper utilities, logging, math functions
│
├── app/                        # Phase 11: Interactive Streamlit UI
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit web app
│   ├── pages/                  # Dedicated multi-page modules
│   ├── components/             # Reusable UI widgets and cards
│   └── static/                 # Custom CSS & design assets
│
├── tests/                      # Unit & integration testing suite
│   ├── __init__.py
│   └── test_setup.py           # Phase 1 setup verification test
│
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation & guidelines
```

---

## 3. Database Architecture

The system utilizes an optimized SQLite database with the following relational tables:

1. **`users`**: User authentication, identifiers, and timestamps.
2. **`news_articles`**: Article metadata (title, abstract, full text, language, category, image URL, published timestamp).
3. **`user_preferences`**: Dynamic language and category preferences stored in JSON format.
4. **`reading_history`**: User reading session history with duration and timestamp tracking.
5. **`user_interactions`**: Explicit and implicit user feedback (clicks, likes, shares, bookmarks).
6. **`recommendation_logs`**: Historical recommendation audit trail containing relevance scores, diversity scores, and generated XAI explanations.

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (compatible with 3.10, 3.11, 3.12, 3.14)
- Pip package manager
- (Optional) NVIDIA GPU with CUDA for accelerated transformer inference

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "d:\Dhanjit sir project"
```

### Step 2: Create and Activate a Python Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. Running the Application & Tests

### Run Setup Verification Tests (Phase 1):
```powershell
python -m unittest tests/test_setup.py
```

### Launch the Streamlit Web Application:
```powershell
streamlit run app/app.py
```

---

## 6. Consolidated Implementation Roadmap (8 Major Phases)

The project development lifecycle has been streamlined into 8 comprehensive, milestone-driven phases:

| Phase | Milestone Title | Status | Technical Scope |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Project Setup & Database Architecture** | **Completed** | Modular project structure, dependency configuration, SQLite relational schema (`users`, `news_articles`, `user_preferences`, `reading_history`, `user_interactions`, `recommendation_logs`). |
| **Phase 2** | **Dataset Collection & Preprocessing** | **Completed** | Unified schema, script-preserving cleaning for English, Hindi, and Assamese, missing-value imputation, deduplication, dataset provenance audit (`data/processed/`). |
| **Phase 3** | **Multilingual Text Representation** | **Completed** | 384-d MiniLM cross-lingual embeddings, 768-d AI4Bharat IndicBERTv2 Assamese encoder with attention-mask-aware mean pooling, gold-standard multilingual benchmark. |
| **Phase 4** | **Image Feature Extraction & Multimodal Fusion** | **Completed** | ResNet-50 visual feature extraction (2048-d), missing image resilience (fallback to text-only), weighted normalized concatenation ($\alpha=0.7, \beta=0.3 \to 2432$-d), projected orthogonal fusion ($384$-d), separate IndicBERT multimodal fusion ($2816$-d), cosine similarity cross-modality analysis. |
| **Phase 5** | **Hybrid & Cross-Lingual Recommendation Engine** | **Completed** | Dynamic user interest profiling (reading history + clicks/likes/bookmarks), Item-Item & User-User Collaborative Filtering (80% sparse interaction matrix), weighted convex hybrid scoring ($\alpha=0.65, \beta=0.35$), full cross-lingual candidate discovery (EN $\leftrightarrow$ HI $\leftrightarrow$ AS), cold-start handling (declared preferences & popularity fallback), diversity re-ranking, and factual XAI explanation generation. |
| **Phase 6** | **Temporal Modelling, Explainable AI & Diversity** | **Completed** | Exponential time decay ($\lambda_{\text{temporal}} = 0.231$, half-life 3.0 days) with long-term interest retention floor, article publication freshness scoring ($\lambda_{\text{fresh}} = 0.139$, half-life 5.0 days), Maximal Marginal Relevance (MMR, $\lambda=0.70$) diversity re-ranking, popularity debiasing ($\gamma=0.10$), multi-signal transparent XAI attribution engine, and ranking stability verification under weight noise, interaction dropout, and modality fallback. |
| **Phase 7** | **Evaluation, Testing & Performance Analysis** | **Completed** | Leak-free hold-out evaluation (Precision, Recall, NDCG@5 = 0.9077, MRR@5 = 0.8750), 5 baseline comparisons, 6 ablation studies, catalog Gini analysis, latency benchmarks (3.8 ms / 263 QPS), 6 interactive figures. |
| **Phase 8** | **Streamlit Dashboard, Integration & Deployment** | **Completed** | Full multi-section Streamlit web app (`app/app.py`): Home Dashboard, News Explorer, Personalized Recommendations, Multilingual & Cross-Lingual discovery, Live User Interactions, XAI Inspector, Analytics charts, and System Info. |

---

## 7. Performance & Offline Evaluation Summary

| Recommendation Paradigm | NDCG@5 | MRR@5 | Novelty@5 | Catalog Coverage (%) | Catalog Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Popularity-Based | $0.5327$ | $0.4583$ | $3.682$ | $43.75\%$ | $0.5625$ |
| Collaborative Filtering | $0.6250$ | $0.5833$ | $3.942$ | $81.25\%$ | $0.4375$ |
| Content-Based (Multimodal) | $0.8577$ | $0.8125$ | $4.451$ | $75.00\%$ | $0.3750$ |
| Hybrid (No Temporal) | $0.9077$ | $0.8750$ | $4.512$ | $81.25\%$ | $0.3520$ |
| **Complete Proposed Framework** | **$0.9077$** | **$0.8750$** | **$4.731$** | **$75.00\%$** | **$0.3125$** |

- **Operational Latency**: **3.80 ms** per recommendation query (**263.3 QPS** throughput).
- **Diversity Gain via MMR**: **+84.89%** category Shannon entropy gain and **-66.67%** redundancy cut.

---

## 8. Deployment Guidelines (Streamlit Community Cloud / Local)

### Local Execution:
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/app.py
```

### Streamlit Community Cloud Deployment:
1. Push the repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the branch to `main`, and specify the main file path:
   `app/app.py`
4. Click **Deploy**. Dependencies from `requirements.txt` will automatically install in the cloud container.

---

## 9. Citation & Academic Purpose
Developed as a final-year B.Tech Computer Science Capstone Project addressing state-of-the-art challenges in Multimodal, Multilingual, and Explainable News Recommendation Systems.

