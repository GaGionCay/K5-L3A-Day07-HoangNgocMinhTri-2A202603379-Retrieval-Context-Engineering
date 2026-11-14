import json
from pathlib import Path
import chromadb

def setup_vector_store():
    # 1. Đường dẫn tới file chunks đã chuẩn hóa ở Phase 1
    chunks_path = Path("data/chunks.json")
    if not chunks_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {chunks_path}. Hãy kiểm tra lại Phase 1.")
        return

    # Đọc nội dung file chunks.json
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Đã đọc thành công {len(chunks)} chunks từ {chunks_path}")

    # 2. Khởi tạo ChromaDB client (lưu dữ liệu cục bộ tại thư mục ./chroma_db của dự án)
    db_path = Path("./chroma_db")
    client = chromadb.PersistentClient(path=str(db_path))

    # Tên collection lưu trữ vector
    collection_name = "day7_corpus"

    # Xóa collection cũ nếu đã tồn tại để làm mới dữ liệu
    try:
        client.delete_collection(collection_name)
        print(f"Đã xóa collection cũ: {collection_name}")
    except Exception:
        pass

    # Tạo collection mới
    collection = client.create_collection(name=collection_name)

    # 3. Trích xuất các trường dữ liệu theo đúng chuẩn schema để đưa vào ChromaDB
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source_id": chunk["source_id"],
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    # 4. Thêm dữ liệu vào Vector Store (ChromaDB tự động embedding bằng mô hình mặc định)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Hoàn tất! Đã nạp thành công {len(chunks)} chunks vào ChromaDB tại thư mục: {db_path.absolute()}")

if __name__ == "__main__":
    setup_vector_store()