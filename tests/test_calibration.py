"""
Test Mean-Centering Calibration and Contrastive Alignment on IndicBERT.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.indicbert_encoder import IndicBERTEncoder

def test_centering():
    pairs_file = Path("data/evaluation/multilingual_eval_pairs.json")
    with open(pairs_file, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)

    encoder = IndicBERTEncoder()

    texts_as = [p["text1"] for p in eval_pairs if p["lang1"] == "as"]
    texts_hi = [p["text2"] for p in eval_pairs if p["lang2"] == "hi"]
    texts_en = [p["text2"] for p in eval_pairs if p["lang2"] == "en"]

    embs_as = encoder.encode(texts_as, normalize_embeddings=False)
    embs_hi = encoder.encode(texts_hi, normalize_embeddings=False)
    embs_en = encoder.encode(texts_en, normalize_embeddings=False)

    mu_as = np.mean(embs_as, axis=0)
    mu_hi = np.mean(embs_hi, axis=0)
    mu_en = np.mean(embs_en, axis=0)

    print("Centering Results:")
    calibrated_results = []
    for p in eval_pairs:
        e1 = encoder.encode(p["text1"], normalize_embeddings=False)
        e2 = encoder.encode(p["text2"], normalize_embeddings=False)

        mu1 = mu_as if p["lang1"] == "as" else (mu_hi if p["lang1"] == "hi" else mu_en)
        mu2 = mu_as if p["lang2"] == "as" else (mu_hi if p["lang2"] == "hi" else mu_en)

        c1 = (e1 - mu1) / (np.linalg.norm(e1 - mu1) + 1e-9)
        c2 = (e2 - mu2) / (np.linalg.norm(e2 - mu2) + 1e-9)

        sim_cal = float(np.dot(c1, c2))
        raw_sim = float(np.dot(e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)))
        calibrated_results.append((p["pair_type"], p["is_equivalent"], raw_sim, sim_cal))
        print(f"{p['id']:15s} [{p['pair_type']}] Equiv={p['is_equivalent']!s:5s} | Raw={raw_sim:.4f} -> Calibrated={sim_cal:+.4f}")

    # Summary by pair type
    print("\nCalibrated Separation Margin:")
    for ptype in ["AS-AS", "AS-HI", "AS-EN", "HI-EN"]:
        sub = [r for r in calibrated_results if r[0] == ptype]
        eq_scores = [r[3] for r in sub if r[1]]
        uneq_scores = [r[3] for r in sub if not r[1]]
        if eq_scores and uneq_scores:
            margin = np.mean(eq_scores) - np.mean(uneq_scores)
            print(f"  {ptype}: Mean Equiv={np.mean(eq_scores):+.4f} | Mean Unrelated={np.mean(uneq_scores):+.4f} | Margin={margin:+.4f}")

if __name__ == "__main__":
    test_centering()
