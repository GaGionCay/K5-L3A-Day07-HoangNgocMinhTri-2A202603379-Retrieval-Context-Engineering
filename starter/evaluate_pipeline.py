"""Script thực nghiệm, chạy benchmark trên Dev/Test set, lưu trace và tính toán Metrics (Phase 5)."""
import json
import time
from pathlib import Path
from typing import Any
from .agentic import HybridRetriever, GeminiController, run_agent


def evaluate_dataset(dataset_path: Path, retriever_method: str, top_k: int, budget: int) -> dict[str, Any]:
    if not dataset_path.exists():
        print(f"Không tìm thấy file dataset: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever(method=retriever_method)
    controller = GeminiController()

    results_trace = []
    total = len(questions)
    correct_answers = 0
    correct_abstains = 0
    total_answerable = 0
    total_unanswerable = 0
    valid_citations_count = 0
    total_citations_emitted = 0

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

        # Đánh giá metrics
        is_correct = False
        if q_type == "answerable":
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

        # Thống kê citation precision
        for c in citations:
            total_citations_emitted += 1
            # Lấy context ids từ sự kiện search cuối cùng nếu có
            events = agent_result.get("events", [])
            last_search = next((e for e in reversed(events) if "context_ids" in e), {})
            available_ctx_ids = last_search.get("context_ids", [])
            if c in available_ctx_ids:
                valid_citations_count += 1

        trace_entry = {
            "id": qid,
            "question": question,
            "type": q_type,
            "gold_chunk_ids": list(gold_chunks),
            "agent_status": status,
            "agent_answer": agent_result.get("answer"),
            "agent_citations": citations,
            "latency_seconds": round(latency, 2),
            "is_correct": is_correct,
            "events": agent_result.get("events", [])
        }
        results_trace.append(trace_entry)
        print(f"[{idx+1}/{total}] ID: {qid} | Type: {q_type} | Status: {status} | Correct: {is_correct} | Latency: {latency:.2f}s")

    # Tổng hợp metrics
    metrics = {
        "dataset": dataset_path.name,
        "method": retriever_method,
        "top_k": top_k,
        "budget": budget,
        "total_questions": total,
        "answerable_count": total_answerable,
        "correct_answers": correct_answers,
        "answer_accuracy": round(correct_answers / total_answerable * 100, 2) if total_answerable > 0 else 0.0,
        "unanswerable_count": total_unanswerable,
        "correct_abstains": correct_abstains,
        "abstention_accuracy": round(correct_abstains / total_unanswerable * 100, 2) if total_unanswerable > 0 else 0.0,
        "citation_precision": round(valid_citations_count / total_citations_emitted * 100, 2) if total_citations_emitted > 0 else 100.0,
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

    # Task 5.2 & 5.3: Chốt cấu hình tối ưu (Ví dụ: phương pháp "hybrid", top_k=4) chạy trên tập Test
    print("================ CHẠY NGHIÊM NGẶT TẬP TEST (Task 5.2 & 5.3) ================")
    evaluate_dataset(test_file, retriever_method="hybrid", top_k=4, budget=3000)


if __name__ == "__main__":
    main()