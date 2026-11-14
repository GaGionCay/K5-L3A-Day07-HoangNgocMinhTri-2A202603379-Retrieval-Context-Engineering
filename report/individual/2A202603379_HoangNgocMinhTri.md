# Task 6.1 — Báo cáo cá nhân

- **MSSV:** 2A202603379
- **Họ và tên:** HoangNgocMinhTri
- **Đề tài:** ReAct-based RAG Agent với Hybrid Retrieval và quản lý Context Budget

> **Phạm vi và tính trung thực của số liệu:** Báo cáo được đối chiếu với mã nguồn, benchmark, ChromaDB và các trace đã lưu trong `starter/results/`. Các chỉ số trong bảng Dev là số liệu thô do `evaluate_pipeline.py` ghi lại, chưa phải đánh giá ngữ nghĩa thủ công. Trace Test hiện có đều lỗi giới hạn quota Gemini trước khi gọi retrieval; vì vậy không thể xem các số 0 trong file trace là kết quả đánh giá chất lượng trên Test. Báo cáo giữ rõ sự khác biệt này và không suy diễn kết quả chưa đo.

## 1. Dữ liệu và benchmark

- **Corpus:** 3 tài liệu dạng văn bản: `system_architecture_guidelines.txt`, `security_and_compliance_policy.txt` và `hr_and_onboarding_handbook.txt`.
- **Chunk/index:** `data/chunks.json` có 8 chunk duy nhất; Chroma collection `day7_corpus` hiện có 8 vector. ChromaDB phiên bản 1.5.9, embedding có 384 chiều. `vector_store_setup.py` không chỉ định embedding function riêng; do đó model embedding cụ thể không được ghim hoặc ghi nhận trong metadata của collection.
- **Page metadata:** trường `page` trong cả 8 chunk hiện là `1`, dù văn bản chunk có các marker `[Page N]`. Vì vậy citation theo chunk ID có thể xác định chunk, nhưng metadata trang chưa đủ chính xác để dùng làm page-level citation.
- **Benchmark:** 30 câu hỏi, 15 Dev và 15 Test. Mỗi split có 12 câu answerable và 3 câu unanswerable; mỗi split có 8 single-hop và 4 multi-hop answerable. Validator báo không trùng câu giữa hai split và toàn bộ gold chunk IDs khớp với 8 ID trong ChromaDB.
- **Đối chiếu yêu cầu benchmark:** tổng quy mô và số câu unanswerable đạt ngưỡng tối thiểu chung. Tuy nhiên Dev chỉ có 12 câu answerable, thấp hơn yêu cầu 15 câu trong `BENCHMARK.md`; cần bổ sung ít nhất 3 câu answerable vào Dev trước khi coi benchmark là hoàn chỉnh.
- **Đánh giá câu trả lời:** trace hiện tại chưa có nhãn chấm tay theo grounded-answer 0–2 hoặc citation support. Chỉ số `answer_accuracy` của pipeline là proxy dựa trên việc citation có trùng ít nhất một `gold_chunk_id`, không kiểm tra câu trả lời có đúng/đủ về ngữ nghĩa hay không.

## 2. System Architecture

### 2.1 ReAct agent loop

`run_agent` bắt đầu bằng hành động `search` khi chưa có lịch sử. Retriever nhận query và trả các chunk kèm ID, nội dung, metadata và score. Agent lưu context cuối cùng cùng `context_ids` vào history rồi gọi Gemini để quyết định bước tiếp theo. Cấu hình trong mã giới hạn tối đa 4 lượt search (`max_searches=4`); câu trả lời chỉ được chấp nhận khi có citations không rỗng, mọi ID đều nằm trong context hiện tại và từng ID xuất hiện dưới dạng `[chunk_id]` trong answer. Nếu thiếu hoặc sai citation, agent từ chối câu trả lời.

Trong các trace Dev đã hoàn thành, mỗi câu có đúng một search event trước khi trả lời hoặc từ chối. Các lỗi 429 xảy ra trước search nên không cung cấp bằng chứng để đánh giá hành vi nhiều lượt.

### 2.2 Hybrid Retrieval pipeline

`HybridRetriever` hỗ trợ ba phương thức:

1. **Sparse:** tách query/document bằng khoảng trắng và đếm số lần xuất hiện của các từ query trong document, sau đó sắp xếp theo score. Đây là lexical term-count retrieval; dù pipeline có một TF-IDF baseline riêng trong `baseline.py`, phương thức `sparse` đang được `evaluate_pipeline.py` gọi **không phải TF-IDF**.
2. **Dense:** gọi `collection.query` trên ChromaDB. Chroma tính embedding query bằng embedding function mặc định tương thích với index; score được chuyển thành `1 / (1 + distance)`.
3. **Hybrid/RRF:** lấy tối đa `top_k` kết quả từ sparse và dense, cộng điểm Reciprocal Rank Fusion theo công thức `1 / (60 + rank)` cho từng danh sách, rồi xếp hạng theo tổng điểm. Hằng số RRF là 60.

### 2.3 Context budget, deduplication và reranking

Sau mỗi lượt search, agent hợp nhất candidate trên toàn bộ các lượt vào map theo chunk ID. Chunk trùng được giữ một lần; nếu cùng ID xuất hiện lại với score cao hơn thì score cao hơn được giữ. Toàn bộ candidate được sắp xếp lại theo score giảm dần trước khi `pack_context` đóng gói context.

`pack_context` thêm tuần tự từng block có ID, source và page trong giới hạn budget tính bằng **ký tự**; block không vừa bị bỏ qua. `run_agent` cập nhật context tích lũy sau mỗi lượt tìm. Đây là global deduplication và reranking theo score; pipeline hiện chưa có semantic reranker độc lập.

## 3. So sánh phương pháp Retrieval trên Dev

Cả ba trace Dev được cấu hình `top_k=3`, `budget=2.000` ký tự và giới hạn 4 lượt search. Các số “đúng” dưới đây là số liệu proxy trong trace, không phải điểm chấm nội dung bằng người.

| Phương pháp | Cấu hình | Answer accuracy theo proxy của pipeline | Abstention accuracy theo pipeline | Citation precision được ghi | Latency trung bình end-to-end |
|---|---|---:|---:|---:|---:|
| Sparse lexical term-count (không phải TF-IDF) | `k=3`, 2.000 ký tự | 8/12 (66,67%); 8 answered, 4 lỗi | 0/3 (0%); cả 3 lỗi | 100% trên 8 citation được phát ra, chỉ kiểm tra ID trong context | 3,74 giây/câu |
| Dense / ChromaDB | `k=3`, 2.000 ký tự | 5/12 (41,67%); 5 answered, 7 lỗi | 2/3 (66,67%); 2 abstain, 1 lỗi | 100% trên 7 citation được phát ra, chỉ kiểm tra ID trong context | 1,87 giây/câu |
| Hybrid / RRF | `k=3`, 2.000 ký tự | 3/12 (25,00%); 3 answered, 9 lỗi | 0/3 (0%); cả 3 lỗi | 100% trên 3 citation được phát ra, chỉ kiểm tra ID trong context | 1,15 giây/câu |

**Cách diễn giải:** Mọi trace lỗi đều ghi `429 RESOURCE_EXHAUSTED`. Pipeline vẫn tính các lỗi như kết quả không đúng trong mẫu số answerable/unanswerable, nên các tỷ lệ trong bảng vừa phản ánh chất lượng vừa bị ảnh hưởng bởi quota. Latency là thời gian end-to-end trung bình trên 15 câu, gồm cả các lần lỗi quota; không nên dùng để kết luận phương thức nào nhanh hơn. Citation precision ở đây chỉ xác nhận ID tồn tại trong context, không xác nhận chunk có hỗ trợ claim.

`evaluate_pipeline.py` mặc định trả citation precision bằng 100% nếu không có citation nào để tính. Vì vậy giá trị này không được diễn giải là chất lượng hoàn hảo; cần báo `0/0 (N/A)` khi không phát ra citation và chấm citation support thủ công trước khi báo strict precision.

## 4. Configuration Analysis

- **Thí nghiệm Dev có thể xác nhận:** ba phương thức retrieval được chạy cùng `k=3` và budget 2.000 ký tự. Sparse có proxy answer score cao nhất trong trace lưu được, nhưng 7/15 câu lỗi và cả 3 câu unanswerable đều lỗi; không thể kết luận sparse tốt hơn hoặc chọn nó làm cấu hình tối ưu.
- **Top-k:** trace Dev dùng `k=3`. Script chỉ định `k=4` cho lần chạy Test, nhưng không có thí nghiệm Dev thay đổi riêng top-k. Vì vậy chưa đo được đánh đổi giữa recall, kích thước context và latency.
- **Character budget:** Dev dùng 2.000 ký tự; cấu hình Test trong script dùng 3.000. Do lần chạy Test lỗi trước retrieval nên chưa có phép so sánh hợp lệ về tác động của budget.
- **Search limit:** `max_searches=4` được đặt trong `run_agent`, nhưng trace hoàn tất chỉ có một search event mỗi câu. Không có thí nghiệm thay đổi giới hạn hoặc kiểm tra chất lượng khi cần nhiều lượt search.
- **Lựa chọn cấu hình:** `evaluate_pipeline.py` hard-code Hybrid `k=4`, budget 3.000 cho Test như cấu hình dự định đánh giá; đây chưa phải cấu hình “optimal” đã được chứng minh trên Dev. Hybrid Dev có 3/12 proxy answers và 12 lỗi quota. Không có sweep top-k/budget/search limit, cũng không có đủ nhãn chấm tay để chốt cấu hình tối ưu.
- **Thử nghiệm tiếp theo:** sau khi bảo đảm quota và bổ sung Dev, so sánh từng yếu tố riêng (ví dụ `k=2/3/4`, budget `1.000/2.000/3.000`, search limit `1/2/4`) trên cùng câu hỏi. Chọn cấu hình theo context evidence rate, grounded answer, abstention, citation support và latency; chỉ sau đó chạy Test một lần.

## 5. Test Evaluation & Error Analysis

### 5.1 Kết quả Test hiện có

Trace được lưu tại `starter/results/trace_test_questions_hybrid_k4.json`, với cấu hình được script chỉ định là Hybrid/RRF, `top_k=4`, budget 3.000 ký tự. Cả **15/15** câu đều có status `error` do `429 RESOURCE_EXHAUSTED`; mỗi lỗi xảy ra trước retrieval, nên tổng số search event là 0 và citation phát ra là 0.

| Chỉ số Test | Kết quả nghiêm ngặt hiện có | Ghi chú |
|---|---:|---|
| Câu hoàn tất inference | 0/15 | 15/15 lỗi quota trước search |
| Answer accuracy (12 answerable) | **N/A** | Không có answer nào được tạo; số 0/12 trong summary thô là lỗi bị tính như câu sai, không phải đánh giá answer |
| Abstention accuracy (3 unanswerable) | **N/A** | Không có quyết định abstain được đánh giá |
| Citation precision | **N/A (0/0)** | Không có citation; giá trị 100% trong summary là mặc định của script khi mẫu số bằng 0 |
| Latency | 0,56 giây/lần chạy tới lỗi (trung bình) | Đây là thời gian thất bại, không phải latency inference thành công |

Do đó chưa có strict Test result và chưa thể khẳng định cấu hình Hybrid `k=4`, budget 3.000 là tối ưu. Khi có quota/API key hợp lệ, cần chạy lại Test sau khi cấu hình đã được chốt trên Dev; đồng thời loại lỗi provider khỏi mẫu số chất lượng và lưu error count riêng.

### 5.2 Phân tích lỗi và giới hạn

1. **Giới hạn quota Gemini:** tất cả lỗi Dev và Test trong các trace đã lưu đều có cùng mã `429 RESOURCE_EXHAUSTED`. Test không tới được bước search. Cần kiểm tra quota/billing, giới hạn tốc độ, và chạy lại sau khi quota được cấp; không nên dùng các lỗi này làm bằng chứng agent trả lời sai hoặc abstain sai.
2. **Proxy answer metric quá yếu:** `evaluate_pipeline.py` xem answer là đúng nếu câu trả lời có status `answered` và citations chứa ít nhất một gold chunk ID. Điều này có thể chấm đúng một câu trả lời sai hoặc thiếu ý; ngược lại, câu trả lời đúng nhưng không trích đúng gold ID sẽ bị chấm sai. Multi-hop cũng được tính đúng với một gold ID bất kỳ, dù câu hỏi có thể cần nhiều chunk.
3. **Citation precision không đánh giá grounding:** metric chỉ so ID với `context_ids`, không đọc claim và evidence. Test trace minh họa lỗi đo lường: không có citation nào nhưng summary vẫn ghi 100% do nhánh mặc định khi mẫu số rỗng.
4. **Benchmark Dev còn thiếu câu answerable:** 12 câu thay vì tối thiểu 15 theo benchmark. Kết quả Dev có độ tin cậy và độ bao phủ hạn chế, đặc biệt cho việc lựa chọn cấu hình.
5. **Page metadata không chính xác:** mọi chunk có `page=1` dù nội dung chứa nhiều marker trang. Citation nên dựa vào chunk ID; nếu cần page citation, cần sửa bước tạo chunk và tái tạo index.
6. **Lexical matching đơn giản:** phương thức sparse hiện đếm từ theo whitespace; không phải TF-IDF/BM25 và chưa xử lý chuẩn hóa token tiếng Việt. Đây là rủi ro thiết kế đã thấy trong mã, chưa có phép đo riêng để định lượng tác động.

## 6. Setup & Execution Guide (PowerShell)

Chạy các lệnh từ thư mục gốc repository trong cùng một PowerShell session:

```powershell
# 1. Tạo và kích hoạt virtual environment (Python 3.11+)
py -3.11 -m venv .venv

# Nếu PowerShell chặn activation, chạy lệnh này rồi kích hoạt lại:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. Cài dependency
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Đặt API key cho session hiện tại; không ghi key vào source control
$env:GOOGLE_API_KEY = '<YOUR_GOOGLE_API_KEY>'

# 4. Kiểm tra benchmark và ChromaDB bằng UTF-8
$env:PYTHONIOENCODING = 'utf-8'
python -m starter.validate_benchmark

# 5. Chạy unit tests
python -m unittest discover -s tests -v

# 6. Chạy toàn bộ pipeline: ba phương thức trên Dev, sau đó Hybrid trên Test
python -m starter.evaluate_pipeline
```

Nếu cần khởi tạo hoặc dựng lại index, có thể chạy `python vector_store_setup.py` trước bước validation. **Lưu ý:** script này xóa collection `day7_corpus` hiện có rồi tạo lại từ `data/chunks.json`; không chạy nếu muốn giữ index hiện tại. Pipeline ghi/ghi đè các trace trong `starter/results/`:

- `trace_dev_questions_sparse_k3.json`
- `trace_dev_questions_dense_k3.json`
- `trace_dev_questions_hybrid_k3.json`
- `trace_test_questions_hybrid_k4.json`

Mỗi câu hiện cần một lần gọi Gemini sau lượt search ban đầu; toàn bộ pipeline có thể tạo khoảng 60 lượt gọi model, chưa kể nếu mở rộng agent để search nhiều vòng. Kiểm tra quota và chi phí trước khi chạy. API key chỉ nên được đặt trong biến môi trường hoặc secret manager; không commit key, trace có dữ liệu nhạy cảm, hay file `.env`.

## 7. Kết luận

Repository đã có ba phương thức retrieval, RRF, context packing có deduplication/reranking toàn cục, benchmark 30 câu và trace cho cả Dev/Test. Tuy nhiên, các trace hiện tại bị quota giới hạn, Dev thiếu câu answerable theo ngưỡng, metric câu trả lời/citation chưa phải đánh giá ngữ nghĩa, và cấu hình Test chưa được chứng minh là tối ưu. Bước tiếp theo là khôi phục quota, hoàn thiện Dev, cải thiện metric bằng chấm grounded answer/citation support thủ công, chạy controlled tuning trên Dev rồi đánh giá Test một lần với cấu hình đã khóa.
