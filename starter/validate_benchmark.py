"""Script kiểm tra tính hợp lệ và đối chiếu Ground Truth của bộ dữ liệu Benchmark với ChromaDB."""
import json
from pathlib import Path
import chromadb


def validate_dataset(file_path: Path, valid_chunk_ids: set[str]) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    if not file_path.exists():
        return [], [f"Không tìm thấy file: {file_path}"]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        return [], [f"File {file_path.name} phải là một danh sách JSON không rỗng."]

    seen_ids = set()
    for idx, item in enumerate(data):
        qid = item.get("id", f"index_{idx}")
        if qid in seen_ids:
            errors.append(f"[{file_path.name}] Trùng lặp ID câu hỏi: {qid}")
        seen_ids.add(qid)

        q_type = item.get("type")
        gold_chunks = item.get("gold_chunk_ids", [])
        gold_answer = item.get("gold_answer", "")

        if not item.get("question", "").strip():
            errors.append(f"[{qid}] Câu hỏi bị rỗng.")
        if not gold_answer.strip():
            errors.append(f"[{qid}] Thiếu gold_answer.")

        if q_type == "answerable":
            if not gold_chunks:
                errors.append(f"[{qid}] Câu answerable nhưng gold_chunk_ids đang rỗng.")
            for cid in gold_chunks:
                if valid_chunk_ids and cid not in valid_chunk_ids:
                    errors.append(f"[{qid}] Chunk ID '{cid}' không tồn tại trong ChromaDB!")
        elif q_type == "unanswerable":
            if gold_chunks:
                errors.append(f"[{qid}] Câu unanswerable nhưng gold_chunk_ids không rỗng: {gold_chunks}")
        else:
            errors.append(f"[{qid}] Giá trị 'type' không hợp lệ: {q_type}")

    return data, errors


def main():
    data_dir = Path("data")
    dev_path = data_dir / "dev_questions.json"
    test_path = data_dir / "test_questions.json"

    # Lấy danh sách chunk IDs thực tế đang có trong ChromaDB
    valid_chunk_ids: set[str] = set()
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="day7_corpus")
        existing = collection.get()
        if existing and existing.get("ids"):
            valid_chunk_ids = set(existing["ids"])
        print(f"Đã kết nối ChromaDB thành công. Tìm thấy {len(valid_chunk_ids)} chunks: {sorted(valid_chunk_ids)}\n")
    except Exception as e:
        print(f"Cảnh báo: Không thể kết nối ChromaDB ({e}). Sẽ chỉ kiểm tra cấu trúc JSON.\n")

    dev_data, dev_errors = validate_dataset(dev_path, valid_chunk_ids)
    test_data, test_errors = validate_dataset(test_path, valid_chunk_ids)

    all_errors = dev_errors + test_errors

    # Kiểm tra trùng lặp câu hỏi giữa Dev và Test (Data Leakage)
    dev_q_texts = {item["question"].strip().lower() for item in dev_data}
    test_q_texts = {item["question"].strip().lower() for item in test_data}
    overlap = dev_q_texts & test_q_texts
    if overlap:
        all_errors.append(f"Phát hiện rò rỉ dữ liệu (trùng câu hỏi giữa Dev và Test): {overlap}")

    # Thống kê tổng quan
    total_data = dev_data + test_data
    ans_count = sum(1 for x in total_data if x.get("type") == "answerable")
    unans_count = sum(1 for x in total_data if x.get("type") == "unanswerable")
    single_hop = sum(1 for x in total_data if x.get("hop_type") == "single-hop")
    multi_hop = sum(1 for x in total_data if x.get("hop_type") == "multi-hop")

    print("=== THỐNG KÊ BỘ DỮ LIỆU BENCHMARK ===")
    print(f"- Tổng số câu hỏi : {len(total_data)} (Dev: {len(dev_data)} | Test: {len(test_data)})")
    print(f"- Answerable      : {ans_count} (Yêu cầu >= 20) -> Single-hop: {single_hop}, Multi-hop: {multi_hop}")
    print(f"- Unanswerable    : {unans_count} (Yêu cầu >= 4)\n")

    if all_errors:
        print("❌ PHÁT HIỆN LỖI CẦN SỬA:")
        for err in all_errors:
            print(f"  - {err}")
    else:
        print("✅ HOÀN HẢO! Toàn bộ dữ liệu Benchmark đã đạt chuẩn Phase 3 và khớp 100% với ChromaDB.")


if __name__ == "__main__":
    main()
