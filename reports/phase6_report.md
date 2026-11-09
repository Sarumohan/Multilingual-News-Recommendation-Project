# Phase 6 Completion Report: Temporal Modelling, Explainable AI and Diversity

**Project**: Advanced Multimodal and Multilingual News Recommendation Framework for Robust, Explainable, and Personalized Content Delivery  
**Languages**: English (`en`), Hindi (`hi`), Assamese (`as`)  
**Phase Completed**: Phase 6 – Temporal Modelling, Explainable AI and Diversity  
**Status**: Verified & Passing (68 Total Project Unit Tests)

---

## 1. Executive Summary

Phase 6 introduces advanced algorithmic intelligence to the news recommendation framework:
1. **Temporal User-Interest Dynamics**: Implements exponential time decay to weight recent user interactions higher while maintaining long-term baseline affinities via a dual-horizon interest model.
2. **Article Freshness Modeling**: Evaluates article publication timestamps to compute publication recency scores, preventing stale articles from dominating.
3. **Transparent Explainable AI (XAI)**: Generates human-readable, multi-signal attributions based on genuine mathematical signals (content similarity, collaborative signals, category affinity, freshness, and language preference).
4. **Diversity-Aware Ranking (MMR)**: Implements Maximal Marginal Relevance to eliminate redundant intra-list recommendations and encourage balanced category coverage.
5. **Popularity-Bias Mitigation**: Introduces smooth logarithmic dampening to prevent disproportionate exposure of head items, allowing niche, high-relevance articles to surface.
6. **Recommendation Stability & Robustness**: Verifies ranking consistency under input weight noise, interaction dropout, and visual modality fallback.

---

## 2. Temporal User-Interest & Article Freshness Architecture

### A. Dual-Horizon User Interest Decay
User preferences evolve dynamically over time. To capture recent surges in reader interest while preventing catastrophic amnesia of persistent long-term tastes, we use a **dual-horizon exponential decay model**:
$$w_i^{\text{temporal}} = w_i^{\text{base}} \times \left[ (1 - \rho) \cdot \exp(-\lambda_{\text{decay}} \cdot \Delta t_i) + \rho \right]$$

- **$\Delta t_i$**: Elapsed time in days between interaction timestamp and the evaluation reference timestamp.
- **$\lambda_{\text{decay}} = \frac{\ln(2)}{3.0 \text{ days}} \approx 0.23105$**: 3.0-day half-life for user interest decay.
- **$\rho = 0.30$**: Long-term interest retention floor parameter (preserves at least 30% baseline weight).
- **Missing / Malformed Timestamps**: Safely parsed via fallback handlers, clamping future timestamps to $\Delta t = 0$ and defaulting unparseable timestamps to neutral weight ($1.0$).

### B. Article Publication Freshness Scoring
$$S_{\text{freshness}}(a) = \exp(-\lambda_{\text{fresh}} \cdot \Delta t_{\text{pub}})$$
- **$\lambda_{\text{fresh}} = \frac{\ln(2)}{5.0 \text{ days}} \approx 0.13863$**: 5.0-day half-life for news freshness.

```text
Interaction Age      | Pure Decay exp(-λΔt) | Dual-Horizon Weight (ρ=0.30) | Article Freshness (5d half-life)
---------------------+----------------------+------------------------------+----------------------------------
Breaking (2 hrs ago) |        0.9809        |            0.9867            |              0.9885
3.0 Days (Half-life) |        0.5000        |            0.6500            |              0.6598
5.0 Days (Fresh half)|        0.3150        |            0.5205            |              0.5000
10.0 Days (Archival) |        0.0992        |            0.3694            |              0.2500
Missing Timestamp    |        1.0000        |            1.0000            |              1.0000 (Safe default)
```

> **Data Authenticity Notice**: The timestamps evaluated are anchored to the project's November 2026 dataset timeframe. In accordance with guidelines, these experiments validate temporal decay mechanics and do not claim to represent organic live-user telemetry.

---

## 3. Explainable AI (XAI) Implementation

Implemented in `src/explainability/explainer.py`, the XAI module decomposes the final ranking score into transparent attribution factors:

```json
{
  "article_id": "art_en_005",
  "title": "National Cricket Championship Delivers Thrilling Final Over Victory",
  "explanation": "Recommended because readers with reading patterns similar to yours also engaged with this Sports story (Collaborative signal: 0.95).",
  "explanation_details": {
    "primary_factor": "collaborative_consensus",
    "signals": {
      "semantic_content_similarity": 0.4336,
      "collaborative_community_signal": 0.9524,
      "user_category_affinity": 0.0,
      "article_freshness": 0.6843
    }
  }
}
```

```json
{
  "article_id": "art_en_003",
  "title": "Space Agency Successfully Completes Deep Space Telescope Calibration",
  "explanation": "Highly aligned with the semantic themes and entities in your reading history (Semantic score: 0.99, English).",
  "explanation_details": {
    "primary_factor": "semantic_content_similarity",
    "signals": {
      "semantic_content_similarity": 0.9885,
      "collaborative_community_signal": 0.0,
      "user_category_affinity": 0.1148,
      "article_freshness": 0.5810
    }
  }
}
```

---

## 4. Diversity-Aware Ranking: Maximal Marginal Relevance (MMR)

Implemented in `src/recommendation/diversity.py`:
$$\text{MMR} = \arg\max_{c_i \in C \setminus S} \left[ \lambda_{\text{mmr}} \cdot \text{Score}(c_i) - (1 - \lambda_{\text{mmr}}) \max_{s_j \in S} \text{Sim}(c_i, s_j) \right]$$

- **$\lambda_{\text{mmr}} = 0.70$**: Balances relevance ($70\%$) with novelty ($30\%$).
- **Similarity Metric**: Cosine similarity on unit-normalized multimodal embeddings.

### Diversification Benchmark
```text
Original Slate (Pure Relevance)        | MMR Diversity Slate (lambda=0.70)
----------------------------------------+-----------------------------------
Rank 1: art_tech_01 (Tech)    Score: 0.95 | Rank 1: art_tech_01 (Tech)     Orig: 1
Rank 2: art_tech_02 (Tech)    Score: 0.94 | Rank 2: art_biz_01  (Business) Orig: 4  [Promoted]
Rank 3: art_tech_03 (Tech)    Score: 0.93 | Rank 3: art_sci_01  (Science)  Orig: 5  [Promoted]
Rank 4: art_biz_01  (Business)Score: 0.85 | Rank 4: art_tech_02 (Tech)     Orig: 2
```

| Metric | Original Slate | MMR Slate | Relative Change |
| :--- | :---: | :---: | :---: |
| **Category Shannon Entropy** | $0.8113$ | **$1.5000$** | **$+84.89\%$ (Diversity Gain)** |
| **Avg Pairwise Similarity (Redundancy)**| $0.5050$ | **$0.1683$** | **$-66.67\%$ (Redundancy Cut)** |
| **Unique Categories in Top-4** | $2$ | **$3$** | $+50.0\%$ |

---

## 5. Popularity Bias Mitigation

Implemented in `src/recommendation/diversity.py#L38`:
$$\text{Score}_{\text{debiased}}(a) = \frac{\text{Score}_{\text{integrated}}(a)}{\left[1 + \log_2(1 + N_{\text{interactions}}(a))\right]^\gamma}$$

With $\gamma = 0.10$:
- $N = 0$ interactions $\to$ Score is unpenalized ($1.0000 \to 1.0000$).
- $N = 5$ interactions $\to$ Score dampened by $11.99\%$ ($1.0000 \to 0.8801$).
- $N = 15$ interactions $\to$ Score dampened by $14.87\%$ ($1.0000 \to 0.8513$).
- $N = 30$ interactions $\to$ Score dampened by $16.34\%$ ($1.0000 \to 0.8366$).

---

## 6. Recommendation Stability Analysis

Implemented in `src/evaluation/stability.py`:

| Stability Stress Test | Nature of Perturbation | Top-5 Jaccard Overlap | Spearman Rank Correlation ($\rho$) | Result |
| :--- | :--- | :---: | :---: | :---: |
| **Interaction Weight Jitter** | Controlled $\pm 10\%$ noise injected into interaction weights | **$1.0000$** | **$1.0000$** | **PASSED** (Highly Stable) |
| **Interaction Dropout** | Single interaction omitted from user profile history | **$0.6667$** | **$1.0000$** | **PASSED** (Robust) |
| **Modality Fallback** | Visual modality dropped $\to$ graceful fallback to text embeddings | **$0.6667$** | **$0.8000$** | **PASSED** (Graceful) |

---

## 7. Integrated Multi-Signal Scoring Pipeline

In `src/recommendation/hybrid_engine.py`, the recommendation pipeline synthesizes:
1. **Base Hybrid Score**: $S_{\text{base}} = \alpha \cdot S_{\text{CB}} + \beta \cdot S_{\text{CF}}$ ($\alpha=0.65, \beta=0.35$).
2. **Freshness Integration**: $S_{\text{integ}} = (1 - w_{\text{fresh}}) \cdot S_{\text{base}} + w_{\text{fresh}} \cdot S_{\text{fresh}}$ ($w_{\text{fresh}}=0.10$).
3. **Popularity Debiasing**: $S_{\text{final}} = S_{\text{integ}} / [1 + \log_2(1 + N_{\text{pop}})]^\gamma$ ($\gamma=0.10$).
4. **MMR Selection**: Re-ranks candidates by $\lambda_{\text{mmr}} S_{\text{final}} - (1-\lambda_{\text{mmr}}) \max \text{Sim}$.
5. **Attribution**: Transparent XAI explanations generated for every slate item.

---

## 8. Test Suite Verification
All 13 unit tests in `tests/test_advanced_intelligence.py` pass cleanly, bringing the total passing project test suite to **68 tests in ~5.6s**.

