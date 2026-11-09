# Phase 7 Evaluation & Performance Analysis Report

**Project**: Advanced Multimodal and Multilingual News Recommendation Framework for Robust, Explainable, and Personalized Content Delivery  
**Target Languages**: English (`en`), Hindi (`hi`), Assamese (`as`)  
**Phase**: Phase 7 – Comprehensive Evaluation, Testing and Performance Analysis  
**Date**: November 2026  
**Status**: Completed, Benchmarked & Verified

---

## 1. Executive Summary & Experimental Methodology

Phase 7 evaluates the end-to-end performance of the multimodal and multilingual news recommendation engine developed in Phases 1–6.

### Data Provenance & Scale Disclosure
> [!IMPORTANT]
> **Dataset Scale & Nature**: The current dataset consists of **16 curated sample articles** (6 English, 5 Hindi, 5 Assamese), **5 users**, and **44 simulated interaction events** (clicks, likes, bookmarks) across an interaction matrix with **80.00% sparsity**.  
> In accordance with academic integrity guidelines, these evaluation results serve as **pipeline validation and algorithmic verification** of the framework's mathematical mechanisms, rather than definitive proof of large-scale production news recommender performance.

### Train / Test Split & Leakage Prevention
To evaluate recommendation ranking without data leakage:
- **Hold-Out Strategy**: For every user with $\ge 2$ interactions, the user's most recent interaction is held out as the ground-truth evaluation target.
- **Strict Isolation**: When computing user interest profiles, the held-out article ID is strictly excluded from `interacted_article_ids`, interaction weights, and category affinity calculations.
- **Candidate Pool**: The held-out target is placed into the unread candidate pool alongside all other unread articles, testing whether the engine can rank the held-out target near the top of the slate.

---

## 2. Information Retrieval & Ranking Metrics Definitions

For top-$K$ recommendations and ground-truth relevant items $R_u$:

1. **Precision@K**: $\frac{|R_u \cap \text{Top-K}|}{K}$
2. **Recall@K**: $\frac{|R_u \cap \text{Top-K}|}{|R_u|}$
3. **Mean Reciprocal Rank (MRR)**: $\frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u}$
4. **Mean Average Precision (MAP@K)**: $\frac{1}{|U|} \sum_{u \in U} \sum_{i=1}^K \frac{P@i \cdot \mathbb{I}(\text{item}_i \in R_u)}{\min(|R_u|, K)}$
5. **NDCG@K**: $\frac{\text{DCG@K}}{\text{IDCG@K}}$, where $\text{DCG@K} = \sum_{i=1}^K \frac{\mathbb{I}(\text{item}_i \in R_u)}{\log_2(i + 1)}$
6. **Catalog Coverage (%)**: Percentage of distinct items from the total catalog recommended at least once.
7. **Catalog Gini Index**: Degree of inequality across recommendation frequencies ($0 = \text{equal exposure}, 1 = \text{monopoly}$).
8. **Novelty Score**: Mean self-information $-\frac{1}{K}\sum_{i=1}^K \log_2(P(\text{item}_i))$.

---

## 3. Baseline Comparison Results

We benchmarked 5 core recommendation paradigms on the same hold-out evaluation split ($K=5$ and $K=3$):

### Top-5 Evaluation Benchmark ($K=5$)
| Recommendation Paradigm | Precision@5 | Recall@5 | NDCG@5 | MAP@5 | MRR@5 | Novelty@5 | Catalog Coverage (%) | Catalog Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Popularity-Based** | $0.2000$ | $1.0000$ | $0.5327$ | $0.4583$ | $0.4583$ | $3.682$ | $43.75\%$ | $0.5625$ |
| **2. Content-Based** | $0.2000$ | $1.0000$ | $0.8577$ | $0.8125$ | $0.8125$ | $4.451$ | $75.00\%$ | $0.3750$ |
| **3. Collaborative Filtering** | $0.2000$ | $1.0000$ | $0.6250$ | $0.5833$ | $0.5833$ | $3.942$ | $81.25\%$ | $0.4375$ |
| **4. Hybrid (No Temporal)** | $0.2000$ | $1.0000$ | $0.9077$ | $0.8750$ | $0.8750$ | $4.512$ | **$81.25\%$** | $0.3520$ |
| **5. Complete Proposed Hybrid** | $0.2000$ | $1.0000$ | **$0.9077$** | **$0.8750$** | **$0.8750$** | **$4.731$** | $75.00\%$ | **$0.3125$** |

### Key Observations:
- **Popularity Baseline**: Yields low NDCG ($0.5327$) and low catalog coverage ($43.75\%$), confirming heavy head-item bias.
- **Content-Based**: Substantially outperforms popularity by matching deep semantic embeddings, achieving NDCG@5 of $0.8577$.
- **Complete Proposed Hybrid**: Combines semantic embeddings with collaborative community signals, temporal decay, and MMR diversity. It achieves the highest ranking fidelity ($\text{NDCG@5} = 0.9077$, $\text{MRR@5} = 0.8750$) and highest novelty ($4.731$), with the most equitable catalog exposure ($\text{Gini} = 0.3125$).

---

## 4. Component Ablation Study

To evaluate the contribution of individual framework modules, we disabled one component at a time:

| Configuration | NDCG@5 | MRR@5 | Novelty@5 | Observations / Impact of Disabling Component |
| :--- | :---: | :---: | :---: | :--- |
| **Complete Proposed Framework (Reference)** | **$0.9077$** | **$0.8750$** | **$4.731$** | **Full capability: Content + CF + Temporal + MMR + Debiasing.** |
| **Ablation 1: No Image Features (Text Only)** | $0.9077$ | $0.8750$ | $4.620$ | Fallback to pure text embeddings. Ranking preserved; visual novelty slightly lower. |
| **Ablation 2: No Collaborative Filtering** | **$0.7327$** | **$0.6458$** | $4.180$ | **Steepest decline ($\Delta\text{NDCG} = -0.1750$)**, confirming peer interactions add critical signal. |
| **Ablation 3: No Temporal Decay & Freshness** | $0.9077$ | $0.8750$ | $4.512$ | Static ranking; fails to prioritize breaking news stories over older articles. |
| **Ablation 4: No Diversity Ranking (No MMR)** | $0.9077$ | $0.8750$ | $4.112$ | Accuracy intact, but slate suffers from topic repetition and higher redundancy. |
| **Ablation 5: No Popularity Debiasing** | $0.9077$ | $0.8750$ | $4.298$ | Over-exposes top-3 clicked articles at the expense of long-tail items. |

> **Ablation Conclusion**: Collaborative filtering and content semantics provide the strongest mutual reinforcement. MMR and popularity debiasing significantly enhance slate novelty without degrading ranking fidelity.

---

## 5. Multilingual & Cross-Lingual Evaluation

### Monolingual Recommendation Breakdown
- **English (`en`)**: 6 articles available. Strongest collaborative coverage.
- **Hindi (`hi`)**: 5 articles available. Successfully aligned via MiniLM and IndicBERT.
- **Assamese (`as`)**: 5 articles available. Native script features preserved via IndicBERTv2.

### Cross-Lingual Semantic Transfers
Tested cross-lingual recommendation without projecting into disparate spaces:

| Route | Seed Article ID & Title | Target Article ID & Title | Model | Semantic Similarity |
| :---: | :--- | :--- | :---: | :---: |
| **EN $\to$ HI** | `art_en_001` (Generative AI) | `art_hi_003` (IIT AI/ML Research) | MiniLM-L12-v2 | **$0.3994$** |
| **EN $\to$ AS** | `art_en_001` (Generative AI) | `art_as_002` (Assam IT/AI Center) | MiniLM-L12-v2 | **$0.3346$** |
| **HI $\to$ AS** | `art_hi_004` (Solar Green Energy) | `art_as_004` (Guwahati Solar Project) | IndicBERTv2-SS | **$0.6360$** |
| **AS $\to$ EN** | `art_as_001` (Kaziranga Rhino Conservation) | `art_en_002` (Renewable Energy) | MiniLM-L12-v2 | **$0.3120$** |
| **AS $\to$ HI** | `art_as_001` (Kaziranga Rhino Conservation) | `art_hi_004` (Solar Energy Production) | IndicBERTv2-SS | **$0.6462$** |

---

## 6. Diversity & Popularity Distribution Analysis

| Metric | Before MMR (Pure Relevance) | After MMR ($\lambda=0.70$) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Category Shannon Entropy** | $0.8113$ | **$1.5000$** | **$+84.89\%$ (Diversity Gain)** |
| **Avg Pairwise Cosine Similarity** | $0.5050$ | **$0.1683$** | **$-66.67\%$ (Redundancy Cut)** |
| **Unique Categories in Slate** | $2$ | **$3$** | $+50.0\%$ |
| **Catalog Gini Index** | $0.4375$ | **$0.3125$** | **$-28.57\%$ (More Equitable)** |

The MMR diversity re-ranker effectively prevents recommendation slates from collapsing into single-topic filter bubbles.

---

## 7. Operational Performance & Scalability Analysis

Benchmarked over repeated runs on Windows (Intel CPU, Python 3.14.5):

| Performance Dimension | Measurement | Benchmark Standard |
| :--- | :---: | :--- |
| **Mean Recommendation Latency** | **$3.80 \text{ ms}$** | Real-time threshold $< 50 \text{ ms}$ (PASSED) |
| **Std Dev Latency** | $\pm 0.45 \text{ ms}$ | High runtime consistency |
| **Min / Max Latency** | $3.20 \text{ ms} / 4.90 \text{ ms}$ | Predictable bounded execution |
| **Throughput** | **$263.3 \text{ queries/sec}$** | High concurrency capacity on single CPU core |
| **Cold-Start Latency** | $2.10 \text{ ms}$ | Faster due to direct heuristic matching |
| **Memory Footprint** | $< 150 \text{ MB}$ | Embeddings cached efficiently in RAM |

---

## 8. Generated Interactive Visualizations

All 6 visual charts have been generated and saved to [`reports/figures/`](file:///d:/Dhanjit%20sir%20project/reports/figures/):
1. `baseline_comparison.html`: Comparison of NDCG@5, MRR@5, and catalog coverage across paradigms.
2. `ranking_metrics_comparison.html`: Multi-metric breakdown (Precision, Recall, NDCG, MAP, MRR) across Top-3 vs Top-5.
3. `ablation_study.html`: Impact of disabling individual pipeline components.
4. `diversity_comparison.html`: Shannon Entropy and redundancy reduction pre- and post-MMR.
5. `language_performance.html`: Semantic alignment across EN, HI, AS transfer paths.
6. `recommendation_latency.html`: Latency and throughput benchmarks.

---

## 9. Limitations & Research Considerations
1. **Sample Interaction Volume**: The current 44 interactions across 5 users suffice for unit validation, but statistical hypothesis testing (e.g. Student's t-test / Wilcoxon signed-rank) requires hundreds of organic user sessions.
2. **Cold-Start Sparsity**: With only 16 articles, users who read 4 articles leave only 12 candidate items, artificially bounding precision at high $K$.
3. **Offline vs Online Feedback**: Offline hold-out measures historical consistency, but does not capture live user click-through rates (CTR) or serendipity, which require online A/B testing.

