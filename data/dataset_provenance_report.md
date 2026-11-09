# Dataset Provenance and Authenticity Audit Report
**Project Title**: Advanced Multimodal and Multilingual News Recommendation Framework for Robust, Explainable, and Personalized Content Delivery  
**Audit Date**: November 2026  
**Auditor**: Machine Learning Architecture Team  
**Scope**: Verification of raw, sample, and processed datasets in `data/`, database integrity, and readiness for downstream recommendation phases.

---

## 1. Executive Summary

This provenance audit was conducted to verify data integrity, distinguish genuine external corpora from development seed datasets, and ensure adherence to scientific standards.

### Key Audit Findings:
1. **Nature of Current 16 Articles**: The current 16 news articles in `data/sample/sample_multilingual_news.csv` and `data/processed/multilingual_news_cleaned.csv` are **manually curated development sample articles**, designed to test multilingual script normalization (English, Hindi, Assamese) and end-to-end recommendation pipelines. They are **not direct scrapes** from publisher APIs.
2. **Source URLs**: All 16 `source_url` fields utilize mock placeholder endpoints under `https://example.com/` (RFC 2606 reserved test domain). They do not point to live publisher articles.
3. **Image URLs**: The `image_url` fields point to Unsplash stock photo CDN identifiers selected for topic relevance (space, clean energy, wildlife, cricket). They are not publisher-hosted editorial press photographs.
4. **MIND Dataset Status**: MIND has **not been downloaded** into the workspace. A production-grade **ingestion adapter** (`load_mind_news_tsv` and `load_mind_behaviors_tsv`) has been implemented and unit-tested in `src/preprocessing/dataset_loader.py` to parse MIND whenever downloaded.
5. **User Interactions**: All 22 user interaction events in `sample_user_interactions.csv` are **synthetically simulated** to model 5 distinct multilingual persona behavioral patterns. They are not real telemetry logs.

---

## 2. Article-by-Article Provenance Audit

| Article ID | Claimed Source | Topic Category | Script / Language | Nature of Text | Source URL Status | Image Nature |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `art_en_001` | *TechDaily* | Technology (AI) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_en_002` | *EcoWorld* | Science (Energy) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_en_003` | *SpaceExplorer* | Science (Space) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_en_004` | *GlobalTech* | Technology (Ethics) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_en_005` | *SportsSphere* | Sports (Cricket) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_en_006` | *FinancialTimes*| Business (CBDC) | Latin (`en`) | Curated development text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_hi_001` | *PIB Hindi* | Science (ISRO) | Devanagari (`hi`) | Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_hi_002` | *Dainik Jagran* | Business (UPI) | Devanagari (`hi`) | Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_hi_003` | *Amar Ujala* | Technology (AI) | Devanagari (`hi`) | Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_hi_004` | *Navbharat* | Science (Solar) | Devanagari (`hi`) | Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_hi_005` | *BBC Hindi* | Sports (Cricket) | Devanagari (`hi`) | Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_as_001` | *Pratidin Time* | Science (Kaziranga)| Eastern Nagari (`as`)| Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_as_002` | *Asomiya Pratidin*| Technology (IT Park)| Eastern Nagari (`as`)| Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_as_003` | *Dainik Janambhumi*| Business (Riverway)| Eastern Nagari (`as`)| Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_as_004` | *Niyomiya Barta*| Science (Solar) | Eastern Nagari (`as`)| Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |
| `art_as_005` | *Pratidin Time* | Business (Tea) | Eastern Nagari (`as`)| Curated regional text | `example.com` (Mock) | Unsplash Stock Photo |

---

## 3. Image URL Resolution and Offline Robustness

- **URL Targets**: All images point to Unsplash CDN URLs (e.g. `https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe`).
- **Network Dependency**: While these URLs are syntactically valid public CDN links, real-world deployment or sandbox environments without active network access cannot fetch external images on the fly.
- **Architectural Safeguard for Phase 4**:
  - The feature extractor will implement a fallback mechanism: if an image URL cannot be fetched due to network isolation or timeout, a deterministic synthetic visual embedding vector (or local cached placeholder image) will be generated to guarantee that Phase 4 vision modeling and Phase 8 multimodal fusion never crash.

---

## 4. Microsoft News Dataset (MIND) Verification

- **Current Status**: **Adapter Implemented; Raw Data Not Downloaded.**
- **Details**:
  - The framework does not currently host the multi-gigabyte `MINDlarge_train.zip` or `MINDsmall_train.zip`.
  - The ingestion interface `NewsDatasetLoader.load_mind_news_tsv()` and `NewsDatasetLoader.load_mind_behaviors_tsv()` is fully coded and tested. It automatically maps MIND's 8-column `news.tsv` and 5-column `behaviors.tsv` into our unified SQLite schema whenever the user places `news.tsv` in `data/raw/`.

---

## 5. User Interaction Behavioral Audit

- **Current Status**: **100% Synthetic Development Seed Data.**
- **Details**:
  - The 22 interaction events in `data/sample/sample_user_interactions.csv` were crafted to provide ground-truth multi-user behavior for developing and debugging recommendation algorithms.
  - Persona mapping:
    - User 1 (`usr_001`): Focuses on English AI & Hindi AI/ISRO news (cross-lingual interest alignment test case).
    - User 2 (`usr_002`): Focuses on Assamese wildlife/tea and English clean energy (cross-lingual environmental test case).
    - User 3 (`usr_003`): Focuses on English and Hindi cricket news (cross-lingual sports test case).
    - User 4 (`usr_004`): Regional bilingual reader (Hindi and Assamese).
    - User 5 (`usr_005`): Cold-start user with single interaction.

---

## 6. Recommended Larger Datasets for Full Experimental Benchmarking

To conduct rigorous empirical evaluation and publishable benchmark comparisons in later phases:

### A. English Modality & Baseline Comparisons
1. **MIND-small (Microsoft News Dataset)**:
   - Size: 50,000 articles, ~150,000 user impression sessions.
   - Purpose: Standard benchmark for comparing against NRMS, NAML, and LSTUR architectures.
   - Licensing: Non-commercial academic research (free via Microsoft Research Open Data).
2. **Adressa Dataset**:
   - Size: Millions of clicks with continuous reading time tracking.
   - Purpose: Validating temporal dwell-time modeling.

### B. Hindi News Modality
1. **BBC Hindi News Dataset**:
   - Size: ~4,335 complete articles categorized into 14 topics.
   - Availability: Publicly archived on GitHub and Kaggle (GPL / Academic).
2. **AI4Bharat IndicCorp (Hindi news subset)**:
   - Size: Millions of news sentences.
   - Purpose: Pretraining or fine-tuning multilingual semantic spaces.

### C. Assamese News Modality
1. **AI4Bharat IndicCorp (Assamese subset)**:
   - Size: Over 1.5 million sentences extracted from Assamese news websites.
   - Availability: AI4Bharat open access under CC-BY / Open Academic License.
2. **Leipzig Corpora Collection (Assamese News)**:
   - Size: 100,000+ Assamese news sentences collected from open web domains.
   - Purpose: Evaluating cross-lingual retrieval and zero-shot categorization.

---

## 7. Audit Conclusion & Phase 3 Readiness

- **Authenticity Status**: The dataset is clearly and transparently documented as a **controlled, manually curated development and testing seed**.
- **Schema Compliance**: 100% compliant with Phase 1 database schema and Phase 2 preprocessing specifications.
- **Next Step**: Phase 3 will generate dense multilingual text embeddings from the preprocessed articles using pretrained transformer checkpoints (`paraphrase-multilingual-MiniLM-L12-v2` / `xlm-roberta-base`) to achieve cross-lingual semantic alignment.

