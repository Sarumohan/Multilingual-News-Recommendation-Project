"""
Unit and Integration Tests for Phase 7 Evaluation and Performance Analysis.
Validates:
1. Metric calculations (Precision, Recall, NDCG, MAP, MRR, Gini, Novelty).
2. Leak-free train/test separation.
3. Baseline comparisons.
4. Ablation study execution.
5. Multilingual and cross-lingual transfers.
6. Operational latency benchmarking.
7. Verification of generated reports and visualizations.
"""

import unittest
from pathlib import Path
from src.evaluation.ranking_metrics import RankingMetrics
from src.evaluation.benchmark_suite import RecommendationBenchmarkSuite

class TestPhase7Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = RankingMetrics()
        cls.suite = RecommendationBenchmarkSuite()
        cls.reports_dir = Path("reports")
        cls.figures_dir = cls.reports_dir / "figures"

    def test_01_precision_and_recall_metrics(self):
        """1. Verify precision and recall calculations with known inputs."""
        actual = ["art_1", "art_2"]
        predicted = ["art_1", "art_3", "art_4", "art_2", "art_5"]

        # K=2: only art_1 is a hit
        self.assertAlmostEqual(self.metrics.precision_at_k(actual, predicted, k=2), 0.50)
        self.assertAlmostEqual(self.metrics.recall_at_k(actual, predicted, k=2), 0.50)

        # K=4: art_1 and art_2 are hits
        self.assertAlmostEqual(self.metrics.precision_at_k(actual, predicted, k=4), 0.50)
        self.assertAlmostEqual(self.metrics.recall_at_k(actual, predicted, k=4), 1.00)

    def test_02_reciprocal_rank_and_ndcg(self):
        """2. Verify MRR and NDCG calculations."""
        actual = ["target_art"]
        # Case A: target at rank 1 -> MRR = 1.0, NDCG = 1.0
        self.assertAlmostEqual(self.metrics.reciprocal_rank(actual, ["target_art", "other"], k=2), 1.0)
        self.assertAlmostEqual(self.metrics.ndcg_at_k(actual, ["target_art", "other"], k=2), 1.0)

        # Case B: target at rank 2 -> MRR = 0.5, NDCG = 1/log2(3) = 0.6309
        self.assertAlmostEqual(self.metrics.reciprocal_rank(actual, ["other", "target_art"], k=2), 0.5)
        self.assertAlmostEqual(self.metrics.ndcg_at_k(actual, ["other", "target_art"], k=2), 1.0 / 1.58496, places=3)

    def test_03_catalog_gini_index(self):
        """3. Verify Gini coefficient for perfect equality vs high inequality."""
        # Perfect equality: every item recommended 5 times
        gini_equal = self.metrics.catalog_gini_index([5, 5, 5, 5, 5])
        self.assertAlmostEqual(gini_equal, 0.0, places=4)

        # Extreme inequality: one item has 100 recommendations, others 0
        gini_unequal = self.metrics.catalog_gini_index([100, 0, 0, 0, 0])
        self.assertGreater(gini_unequal, 0.70)

    def test_04_novelty_score(self):
        """4. Verify self-information novelty score."""
        probs = {"head_item": 0.50, "tail_item": 0.01}
        # Tail item should have strictly higher novelty (-log2(0.01) > -log2(0.5))
        nov_head = self.metrics.novelty_score(["head_item"], probs)
        nov_tail = self.metrics.novelty_score(["tail_item"], probs)
        self.assertGreater(nov_tail, nov_head)

    def test_05_train_test_split_leak_prevention(self):
        """5. Verify held-out targets are strictly excluded from profile training data."""
        self.assertGreater(len(self.suite.held_out_targets), 0)
        for uid, held_out in self.suite.held_out_targets.items():
            # Build profile excluding held out
            prof = self.suite.engine.profile_modeler.build_user_profile(
                user_id=uid,
                exclude_article_ids={held_out}
            )
            # The held-out item must NOT be in the profile's training interactions
            self.assertNotIn(held_out, prof.interacted_article_ids)
            self.assertNotIn(held_out, prof.interaction_scores)

    def test_06_baseline_comparisons_execution(self):
        """6. Verify execution of all 5 baselines with valid metrics."""
        baselines = self.suite.run_baseline_comparisons(k=5)
        self.assertEqual(len(baselines), 5)
        for b in baselines:
            self.assertIn("config_name", b)
            self.assertIn("ndcg@5", b)
            self.assertIn("mrr@5", b)
            self.assertIn("catalog_coverage_pct", b)
            self.assertTrue(0.0 <= b["ndcg@5"] <= 1.0)
            self.assertTrue(0.0 <= b["catalog_coverage_pct"] <= 100.0)

    def test_07_ablation_study_execution(self):
        """7. Verify execution of all 6 component ablation configurations."""
        ablations = self.suite.run_ablation_study(k=5)
        self.assertEqual(len(ablations), 6)
        ref_ndcg = ablations[0]["ndcg@5"]
        for a in ablations:
            self.assertIn("config_name", a)
            self.assertTrue(0.0 <= a["ndcg@5"] <= 1.0)

    def test_08_multilingual_and_cross_lingual_evaluation(self):
        """8. Verify multilingual breakdowns and cross-lingual transfers."""
        res = self.suite.run_multilingual_evaluation(k=3)
        self.assertIn("monolingual_breakdown", res)
        self.assertIn("cross_lingual_transfers", res)

        breakdown = res["monolingual_breakdown"]
        for lang in ["en", "hi", "as"]:
            self.assertIn(lang, breakdown)
            self.assertGreater(breakdown[lang]["total_articles"], 0)

        transfers = res["cross_lingual_transfers"]
        self.assertGreater(len(transfers), 0)
        for t in transfers:
            self.assertTrue(0.0 <= t["semantic_similarity"] <= 1.0)

    def test_09_performance_benchmark(self):
        """9. Verify latency benchmarking returns positive, sub-50ms timings."""
        perf = self.suite.run_performance_benchmarks(num_runs=5)
        self.assertIn("mean_recommendation_latency_ms", perf)
        self.assertIn("throughput_queries_per_sec", perf)
        self.assertLess(perf["mean_recommendation_latency_ms"], 50.0)
        self.assertGreater(perf["throughput_queries_per_sec"], 20.0)

    def test_10_reports_and_figures_generated(self):
        """10. Verify that Phase 7 report and all 6 visualization HTML files exist."""
        report_file = self.reports_dir / "phase7_evaluation_report.md"
        self.assertTrue(report_file.exists(), "Missing phase7_evaluation_report.md")

        required_figures = [
            "baseline_comparison.html",
            "ranking_metrics_comparison.html",
            "ablation_study.html",
            "diversity_comparison.html",
            "language_performance.html",
            "recommendation_latency.html"
        ]
        for fig_name in required_figures:
            fig_path = self.figures_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Missing figure: {fig_name}")


if __name__ == "__main__":
    unittest.main()

