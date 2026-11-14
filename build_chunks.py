import json
from pathlib import Path

def build_corpus_and_chunks():
    # 1. Đọc file corpus manifest từ starter
    manifest_path = Path("starter/data/corpus.json")
    if not manifest_path.exists():
        print(f"Không tìm thấy file manifest tại {manifest_path}")
        return

    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    all_chunks = []
    
    # 2. Duyệt qua từng tài liệu trong manifest để đọc nội dung file .txt tương ứng
    for doc in corpus_manifest:
        doc_id = doc.get("document_id")
        source_file = doc.get("source")
        # Đường dẫn tới file text gốc (thay đổi tùy theo cấu trúc thư mục thực tế của starter)
        txt_path = Path("starter/data/corpus") / source_file
        
        if not txt_path.exists():
            print(f"Cảnh báo: Không tìm thấy file text tại {txt_path}, bỏ qua.")
            continue
            
        content = txt_path.read_text(encoding="utf-8")
        
        # 3. Chia nhỏ văn bản thành các chunk (ví dụ mỗi chunk khoảng 500 ký tự)
        chunk_size = 500
        for idx, start in enumerate(range(0, len(content), chunk_size)):
            chunk_text = content[start:start + chunk_size].strip()
            if not chunk_text:
                continue
                
            chunk_item = {
                "id": f"{doc_id}_chunk_{idx}",
                "text": chunk_text,
                "source_id": doc_id,
                "source": source_file,
                "page": 1, # Mặc định trang 1 hoặc bạn có thể phân trang nếu file text có đánh dấu
                "metadata": doc.get("metadata", {})
            }
            all_chunks.append(chunk_item)

    # 4. Ghi kết quả ra thư mục data/chunks.json
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_output_path = output_dir / "chunks.json"
    chunks_output_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã tạo thành công {len(all_chunks)} chunks tại {chunks_output_path}")

    # 5. Tạo file questions.json mẫu (hoặc bạn copy từ starter nếu có sẵn)
    questions_output_path = output_dir / "questions.json"
    if not questions_output_path.exists():
        sample_questions = [
            {
                "id": "q1",
                "split": "dev",
                "question": "Tiêu chuẩn thiết kế kiến trúc hệ thống phân tán là gì?",
                "expected_answer": "Xem chi tiết trong tài liệu hệ thống.",
                "gold": [
                    {
                        "source_id": "doc_01",
                        "page": 1,
                        "evidence": "Tiêu chuẩn thiết kế kiến trúc hệ thống phân tán và microservices."
                    }
                ]
            }
        ]
        questions_output_path.write_text(json.dumps(sample_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Đã tạo file câu hỏi mẫu tại {questions_output_path}")

if __name__ == "__main__":
    build_corpus_and_chunks()