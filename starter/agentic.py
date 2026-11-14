"""Agent loop optimized with 3 Retrieval methods (Dense, Sparse, Hybrid/RRF) and Fixed Context Budget."""
from typing import Any, Protocol
import json
import re
from pathlib import Path
import chromadb
from google import genai
from .baseline import pack_context
import math
from collections import Counter


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action."""
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the vector store or corpus."""
        ...


class HybridRetriever:
    """Task 4.1: Tích hợp 3 phương pháp Retrieval (Sparse, Dense, Hybrid RRF)."""
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "day7_corpus", method: str = "hybrid"):
        self.method = method  # "sparse", "dense", hoặc "hybrid"
        self.client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = None

    def _sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Sparse Retrieval dự phòng bằng từ khóa trên collection."""
        if not self.collection:
            return []
        all_data = self.collection.get()
        if not all_data or not all_data.get("ids"):
            return []
        
        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        
        query_tokens = set(query.lower().split())
        scored = []
        for cid, doc, meta in zip(ids, docs, metas):
            doc_tokens = doc.lower().split()
            doc_counter = Counter(doc_tokens)
            score = sum(doc_counter[token] for token in query_tokens)
            if score > 0:
                scored.append({
                    "id": cid, "text": doc, "source_id": meta.get("source_id"),
                    "source": meta.get("source"), "page": meta.get("page", 1),
                    "metadata": meta, "score": float(score)
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _dense_search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Dense Retrieval trực tiếp từ ChromaDB."""
        if not self.collection:
            return []
        results = self.collection.query(query_texts=[query], n_results=top_k, where=filters)
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
                formatted.append({
                    "id": cid,
                    "text": doc,
                    "source_id": meta.get("source_id"),
                    "source": meta.get("source"),
                    "page": meta.get("page", 1),
                    "metadata": meta,
                    "score": float(1.0 / (1.0 + dist))
                })
        return formatted

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if self.method == "sparse":
            return self._sparse_search(query, top_k=top_k)
            
        elif self.method == "dense":
            return self._dense_search(query, filters=filters, top_k=top_k)

        else:
            # Method 3: Hybrid Retrieval kết hợp Sparse + Dense dùng RRF (Tránh đệ quy vô hạn)
            sparse_res = self._sparse_search(query, top_k=top_k)
            dense_res = self._dense_search(query, filters=filters, top_k=top_k)
            
            rrf_scores = {}
            chunk_map = {}
            k_constant = 60  # Hằng số chuẩn cho RRF

            for rank, chunk in enumerate(sparse_res):
                cid = chunk["id"]
                chunk_map[cid] = chunk
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)

            for rank, chunk in enumerate(dense_res):
                cid = chunk["id"]
                chunk_map[cid] = chunk
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)

            sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            hybrid_results = []
            for cid in sorted_cids[:top_k]:
                item = chunk_map[cid]
                item["score"] = rrf_scores[cid]
                hybrid_results.append(item)
            return hybrid_results


class GeminiController:
    """Adapter thực tế dùng Google Gemini điều khiển ReAct agent."""
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.client = genai.Client()
        self.model_name = model_name

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {
                "action": "search",
                "query": question
            }

        context_ids = list(dict.fromkeys(
            chunk_id
            for h in history
            for chunk_id in h.get("context_ids", [])
            if isinstance(chunk_id, str) and chunk_id
        ))
        available_ids = json.dumps(context_ids, ensure_ascii=False)
        citation_example = (
            f"For example, use [{context_ids[0]}] in the answer and "
            f'"{context_ids[0]}" in citations.'
            if context_ids else "No chunk IDs are available in the context."
        )

        system_instruction = (
            "You answer the user's question EXCLUSIVELY from the retrieved context. "
            f"The only valid chunk IDs are {available_ids}. Copy them exactly.\n\n"
            "Return one JSON object and nothing outside it, using this schema:\n"
            "{\"action\":\"answer\",\"answer\":\"... [exact_chunk_id] ...\",\"citations\":[\"exact_chunk_id\"]}\n"
            f"{citation_example}\n\n"
            "Citation rules:\n"
            "- Put each supporting chunk ID directly in the answer as the exact bracketed token "
            "[chunk_id], next to every factual claim it supports.\n"
            "- In 'citations', return the same IDs as bare strings; do not include brackets there.\n"
            "- Include no citation ID that is not present in the retrieved context, and do not "
            "omit an ID used in the answer text.\n"
            "- If the context supports an answer, use action 'answer' and cite it. If it does not, "
            "state that there is not enough evidence and use an empty citations array."
        )

        history_str = "Retrieved context from search:\n"
        for h in history:
            context = h.get("context", "")
            history_str += f"{context}\n\n"

        prompt = f"Question: {question}\n\n{history_str}\nProvide the final answer in JSON format:"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                'system_instruction': system_instruction,
                'temperature': 0.1,
                'response_mime_type': 'application/json'
            }
        )
        text = response.text.strip()
        
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
            raise ValueError("Gemini response must be an object with a string answer")

        answer = parsed["answer"]
        cited_ids = re.findall(r"\[([^\[\]]+)\]", answer)
        if any(chunk_id not in context_ids for chunk_id in cited_ids):
            raise ValueError("Gemini response cited a chunk outside the retrieved context")
        citations = list(dict.fromkeys(cited_ids))
        return {"action": "answer", "answer": answer, "citations": citations}


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 1500) -> dict[str, Any]:
    """Task 4.2: Khắc phục Technical Debt về Context Budget bằng Deduplication và Re-ranking toàn cục."""
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    
    history: list[dict[str, Any]] = []
    all_candidates: dict[str, dict[str, Any]] = {}  # Khử trùng lặp theo chunk['id'] qua các vòng tìm kiếm
    events: list[dict[str, Any]] = []

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            
            results = retriever.search(query, filters=filters, top_k=top_k)
            
            # Global Deduplication & Score Update
            for r in results:
                cid = r['id']
                if cid not in all_candidates or r.get('score', 0) > all_candidates[cid].get('score', 0):
                    all_candidates[cid] = r

            # Global Re-ranking dựa trên score giảm dần trước khi đóng gói budget
            ranked_candidates = sorted(all_candidates.values(), key=lambda x: x.get('score', 0), reverse=True)
            
            context, ids = pack_context(ranked_candidates, budget=budget)
            
            events.append({
                'event': 'search', 'query': query, 'filter': filters, 'results': results,
                'context': context, 'context_ids': ids
            })
            history = [{
                'tool': 'search', 
                'queries': [e['query'] for e in events if e['event'] == 'search'],
                'context': context, 
                'context_ids': ids
            }]
            
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            
            current_context_ids = history[-1].get('context_ids', []) if history else []
            valid = bool(answer.strip() and citations and all(i in current_context_ids and f'[{i}]' in answer for i in citations))
            
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('action phải là search hoặc answer')
            
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}


if __name__ == '__main__':
    retriever = HybridRetriever(method="hybrid")
    controller = GeminiController()
    
    test_question = "Tiêu chuẩn thiết kế kiến trúc hệ thống phân tán là gì?"
    print(f"Đang chạy kiểm thử với câu hỏi: '{test_question}' (Hybrid Method)\n")
    
    result = run_agent(test_question, controller, retriever, top_k=4, budget=4000)
    print("--- KẾT QUẢ AGENT ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))