"""Script thực nghiệm, chạy benchmark trên Dev/Test set, lưu trace và tính toán Metrics (Phase 5)."""
import json
import time
from pathlib import Path
from typing import Any
from .agentic import HybridRetriever, GeminiController, run_agent
from .retrieval_metrics import (
    calculate_mrr,
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_context_evidence_rate,
    evaluate_retrieval,
)


CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "retriever_method": "hybrid",
    "top_k": 4,
    "budget": 3000,
    "max_searches": 4,
    "model_name": "gemini-3.5-flash",
}


def load_frozen_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return data.get("production_config", DEFAULT_CONFIG)
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def evaluate_dataset(dataset_path: Path, retriever_method: str | None = None, top_k: int | None = None, budget: int | None = None) -> dict[str, Any]:
    cfg = load_frozen_config()
    retriever_method = retriever_method or cfg["retriever_method"]
    top_k = top_k or cfg["top_k"]
    budget = budget or cfg["budget"]

    if not dataset_path.exists():
        print(f"Không tìm thấy file dataset: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever(method=retriever_method)
    controller = GeminiController(model_name=cfg.get("model_name", "gemini-3.5-flash"))

    results_trace = []
    total = len(questions)
    correct_answers = 0
    correct_abstains = 0
    total_answerable = 0
    total_unanswerable = 0
    provider_errors_count = 0
    valid_citations_count = 0
    total_citations_emitted = 0

    # Task 3.4: Accumulators for retrieval evaluation metrics across answerable questions
    retrieval_mrr_sum = 0.0
    retrieval_hit1_sum = 0
    retrieval_hit3_sum = 0
    retrieval_recall1_sum = 0.0
    retrieval_recall3_sum = 0.0
    retrieval_evidence_rate_sum = 0.0
    eval_retrieval_count = 0

    print(f"\n--- Đang chạy đánh giá trên: {dataset_path.name} | Method: {retriever_method.upper()} (top_k={top_k}, budget={budget}) ---")

    for idx, item in enumerate(questions):
        qid = item.get("id")
        question = item.get("question")
        q_type = item.get("type") # "answerable" hoặc "unanswerable"
        gold_chunks = set(item.get("gold_chunk_ids", []))

        start_time = time.time()
        try:
            agent_result = run_agent(question, controller, retriever, top_k=top_k, budget=budget)
        except Exception as e:
            agent_result = {
                "status": "error",
                "answer": str(e),
                "citations": [],
                "events": [{"event": "exception", "error": str(e)}]
            }
        latency = time.time() - start_time

        status = agent_result.get("status")
        citations = agent_result.get("citations", [])

        # Đánh giá metrics và tách biệt Provider / System Errors (Task 4.2)
        is_provider_error = (status == "error")
        is_correct = False

        if is_provider_error:
            provider_errors_count += 1
        elif q_type == "answerable":
            total_answerable += 1
            # Kiểm tra xem agent có trả lời được và có chứa ít nhất một gold chunk không
            if status == "answered" and any(c in gold_chunks for c in citations):
                correct_answers += 1
                is_correct = True
        elif q_type == "unanswerable":
            total_unanswerable += 1
            # Kiểm tra xem agent có từ chối đúng hay không
            if status == "abstain" or agent_result.get("answer") == "Không đủ bằng chứng để trả lời.":
                correct_abstains += 1
                is_correct = True

        # Thống kê citation precision & retrieval metrics
        events = agent_result.get("events", [])
        last_search = next((e for e in reversed(events) if "context_ids" in e), {})
        available_ctx_ids = last_search.get("context_ids", [])

        for c in citations:
            total_citations_emitted += 1
            if c in available_ctx_ids:
                valid_citations_count += 1

        # Retrieval metrics computation for queries with gold chunks
        query_retrieval_metrics = None
        if gold_chunks:
            q_mrr = calculate_mrr(available_ctx_ids, list(gold_chunks))
            q_hit1 = calculate_hit_at_k(available_ctx_ids, list(gold_chunks), k=1)
            q_hit3 = calculate_hit_at_k(available_ctx_ids, list(gold_chunks), k=3)
            q_recall1 = calculate_recall_at_k(available_ctx_ids, list(gold_chunks), k=1)
            q_recall3 = calculate_recall_at_k(available_ctx_ids, list(gold_chunks), k=3)
            q_evidence_rate = calculate_context_evidence_rate(available_ctx_ids, list(gold_chunks))

            retrieval_mrr_sum += q_mrr
            retrieval_hit1_sum += q_hit1
            retrieval_hit3_sum += q_hit3
            retrieval_recall1_sum += q_recall1
            retrieval_recall3_sum += q_recall3
            retrieval_evidence_rate_sum += q_evidence_rate
            eval_retrieval_count += 1

            query_retrieval_metrics = {
                "mrr": round(q_mrr, 4),
                "hit@1": q_hit1,
                "hit@3": q_hit3,
                "recall@1": round(q_recall1, 4),
                "recall@3": round(q_recall3, 4),
                "context_evidence_rate": round(q_evidence_rate, 4),
            }

        trace_entry = {
            "id": qid,
            "question": question,
            "type": q_type,
            "gold_chunk_ids": list(gold_chunks),
            "retrieved_chunk_ids": available_ctx_ids,
            "retrieval_metrics": query_retrieval_metrics,
            "agent_status": status,
            "is_provider_error": is_provider_error,
            "agent_answer": agent_result.get("answer"),
            "agent_citations": citations,
            "latency_seconds": round(latency, 2),
            "is_correct": is_correct,
            "events": agent_result.get("events", [])
        }
        results_trace.append(trace_entry)
        retrieval_info = f" | MRR: {query_retrieval_metrics['mrr']:.2f} | Hit@3: {query_retrieval_metrics['hit@3']}" if query_retrieval_metrics else ""
        print(f"[{idx+1}/{total}] ID: {qid} | Type: {q_type} | Status: {status} | Correct: {is_correct}{retrieval_info} | Latency: {latency:.2f}s")

    # Tổng hợp metrics
    metrics = {
        "dataset": dataset_path.name,
        "method": retriever_method,
        "top_k": top_k,
        "budget": budget,
        "total_questions": total,
        "provider_errors": provider_errors_count,
        "answerable_evaluated": total_answerable,
        "correct_answers": correct_answers,
        "answer_accuracy": round(correct_answers / total_answerable * 100, 2) if total_answerable > 0 else 0.0,
        "unanswerable_evaluated": total_unanswerable,
        "correct_abstains": correct_abstains,
        "abstention_accuracy": round(correct_abstains / total_unanswerable * 100, 2) if total_unanswerable > 0 else 0.0,
        "citation_precision": round(valid_citations_count / total_citations_emitted * 100, 2) if total_citations_emitted > 0 else 100.0,
        # Task 3.4: Retrieval Quality Evaluation Stats
        "retrieval_evaluated_queries": eval_retrieval_count,
        "retrieval_mrr": round(retrieval_mrr_sum / eval_retrieval_count, 4) if eval_retrieval_count > 0 else 0.0,
        "retrieval_hit@1": round(retrieval_hit1_sum / eval_retrieval_count * 100, 2) if eval_retrieval_count > 0 else 0.0,
        "retrieval_hit@3": round(retrieval_hit3_sum / eval_retrieval_count * 100, 2) if eval_retrieval_count > 0 else 0.0,
        "retrieval_recall@1": round(retrieval_recall1_sum / eval_retrieval_count, 4) if eval_retrieval_count > 0 else 0.0,
        "retrieval_recall@3": round(retrieval_recall3_sum / eval_retrieval_count, 4) if eval_retrieval_count > 0 else 0.0,
        "retrieval_context_evidence_rate": round(retrieval_evidence_rate_sum / eval_retrieval_count, 4) if eval_retrieval_count > 0 else 0.0,
    }

    # Lưu trace và metrics ra file JSON trong thư mục starter/results/
    results_dir = Path("starter/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"trace_{dataset_path.stem}_{retriever_method}_k{top_k}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "traces": results_trace}, f, ensure_ascii=False, indent=2)

    print(f"\n=== KẾT QUẢ TỔNG HỢP ({dataset_path.name}) ===")
    for k, v in metrics.items():
        print(f" - {k}: {v}")
    print(f"Đã lưu trace chi tiết tại: {output_file}\n")
    return metrics


def main():
    data_dir = Path("data")
    dev_file = data_dir / "dev_questions.json"
    test_file = data_dir / "test_questions.json"

    # Task 5.1: Thực nghiệm có kiểm soát trên tập Dev với 3 phương pháp Retrieval
    print("================ THỰC NGHIỆM TẬP DEV (Task 5.1) ================")
    for method in ["sparse", "dense", "hybrid"]:
        evaluate_dataset(dev_file, retriever_method=method, top_k=3, budget=2000)

    # Task 5.2 & 5.3: Chốt cấu hình tối ưu frozen từ config.json chạy trên tập Test
    print("================ CHẠY NGHIÊM NGẶT TẬP TEST (Task 5.2 & 5.3 - FROZEN CONFIG) ================")
    cfg = load_frozen_config()
    evaluate_dataset(test_file, retriever_method=cfg["retriever_method"], top_k=cfg["top_k"], budget=cfg["budget"])


if __name__ == "__main__":
    main()