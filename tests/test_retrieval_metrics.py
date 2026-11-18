import unittest
from starter.retrieval_metrics import (
    calculate_mrr,
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_context_evidence_rate,
    evaluate_retrieval,
)
from starter.agentic import HybridRetriever


class RetrievalMetricsTests(unittest.TestCase):
    def test_calculate_mrr(self):
        # First position hit
        self.assertEqual(calculate_mrr(["c1", "c2", "c3"], ["c1"]), 1.0)
        # Second position hit
        self.assertEqual(calculate_mrr(["c2", "c1", "c3"], ["c1"]), 0.5)
        # Third position hit
        self.assertAlmostEqual(calculate_mrr(["c2", "c3", "c1"], ["c1"]), 1.0 / 3)
        # Multiple gold chunks: takes first hit
        self.assertEqual(calculate_mrr(["c2", "c1", "c3"], ["c3", "c1"]), 0.5)
        # No hit
        self.assertEqual(calculate_mrr(["c2", "c3"], ["c1"]), 0.0)
        # Empty inputs
        self.assertEqual(calculate_mrr([], ["c1"]), 0.0)
        self.assertEqual(calculate_mrr(["c1"], []), 0.0)

    def test_calculate_hit_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        self.assertEqual(calculate_hit_at_k(retrieved, ["c1"], k=1), 1)
        self.assertEqual(calculate_hit_at_k(retrieved, ["c2"], k=1), 0)
        self.assertEqual(calculate_hit_at_k(retrieved, ["c2"], k=2), 1)
        self.assertEqual(calculate_hit_at_k(retrieved, ["c5"], k=4), 0)
        self.assertEqual(calculate_hit_at_k(retrieved, ["c5"], k=0), 0)
        self.assertEqual(calculate_hit_at_k([], ["c1"], k=3), 0)

    def test_calculate_recall_at_k(self):
        retrieved = ["c1", "c2", "c3"]
        gold = ["c1", "c3", "c4"]
        # k=1: retrieves ['c1'], 1 out of 3 golds
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, gold, k=1), 1.0 / 3)
        # k=2: retrieves ['c1', 'c2'], 1 out of 3 golds
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, gold, k=2), 1.0 / 3)
        # k=3: retrieves ['c1', 'c2', 'c3'], 2 out of 3 golds
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, gold, k=3), 2.0 / 3)
        # Empty gold
        self.assertEqual(calculate_recall_at_k(retrieved, [], k=3), 0.0)

    def test_calculate_context_evidence_rate(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        gold = ["c1", "c3"]
        # In all 4: 2 gold chunks -> 2/4 = 0.5
        self.assertEqual(calculate_context_evidence_rate(retrieved, gold), 0.5)
        # In top 2: ['c1', 'c2'] -> 1 gold chunk -> 1/2 = 0.5
        self.assertEqual(calculate_context_evidence_rate(retrieved, gold, k=2), 0.5)
        # In top 1: ['c1'] -> 1/1 = 1.0
        self.assertEqual(calculate_context_evidence_rate(retrieved, gold, k=1), 1.0)
        # In top 2 with no gold
        self.assertEqual(calculate_context_evidence_rate(["c2", "c4"], gold), 0.0)
        # Empty retrieved
        self.assertEqual(calculate_context_evidence_rate([], gold), 0.0)

    def test_evaluate_retrieval(self):
        res = evaluate_retrieval(["c1", "c2", "c3"], ["c2", "c4"], k_values=(1, 3))
        self.assertEqual(res["mrr"], 0.5)
        self.assertEqual(res["hit@1"], 0)
        self.assertEqual(res["hit@3"], 1)
        self.assertEqual(res["recall@1"], 0.0)
        self.assertEqual(res["recall@3"], 0.5)
        self.assertAlmostEqual(res["context_evidence_rate"], 1.0 / 3)


class HybridRetrieverRRFTests(unittest.TestCase):
    def test_rrf_scoring_and_tracking(self):
        retriever = HybridRetriever(method="hybrid")
        # Mock sparse and dense results to test exact RRF formula
        retriever._sparse_search = lambda query, top_k: [
            {"id": "doc_a", "score": 2.5, "text": "Doc A"},
            {"id": "doc_b", "score": 1.5, "text": "Doc B"},
        ]
        retriever._dense_search = lambda query, filters, top_k: [
            {"id": "doc_b", "score": 0.9, "text": "Doc B"},
            {"id": "doc_c", "score": 0.8, "text": "Doc C"},
        ]

        results = retriever.search("test query", top_k=3)
        self.assertEqual(len(results), 3)

        # Expected RRF scores with K = 60:
        # doc_a: sparse_rank=1 -> 1 / (60 + 1) = 1/61 (~0.016393)
        # doc_b: sparse_rank=2, dense_rank=1 -> 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61 (~0.032522)
        # doc_c: dense_rank=2 -> 1 / (60 + 2) = 1/62 (~0.016129)
        # Therefore rank order must be: doc_b, doc_a, doc_c
        self.assertEqual(results[0]["id"], "doc_b")
        self.assertEqual(results[1]["id"], "doc_a")
        self.assertEqual(results[2]["id"], "doc_c")

        expected_b_score = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
        self.assertAlmostEqual(results[0]["score"], expected_b_score)
        self.assertIn("rrf_details", results[0])
        self.assertEqual(results[0]["rrf_details"]["k"], 60)
        self.assertEqual(results[0]["rrf_details"]["sparse_rank"], 2)
        self.assertEqual(results[0]["rrf_details"]["dense_rank"], 1)

        expected_a_score = 1.0 / (60 + 1)
        self.assertAlmostEqual(results[1]["score"], expected_a_score)
        self.assertEqual(results[1]["rrf_details"]["sparse_rank"], 1)
        self.assertIsNone(results[1]["rrf_details"]["dense_rank"])


if __name__ == "__main__":
    unittest.main()
