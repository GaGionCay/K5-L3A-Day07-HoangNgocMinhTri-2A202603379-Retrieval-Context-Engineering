"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
from typing import Any, Protocol
import json
import re
from pathlib import Path
import chromadb
from google import genai
from .baseline import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


class ChromaRetriever:
    """Adapter thực tế kết nối ChromaDB Vector Store (từ Task 2.1)."""
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "day7_corpus"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = None

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.collection:
            return []
        
        where_clause = filters if filters else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )
        
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


class GeminiController:
    """Adapter thực tế dùng Google Gemini điều khiển ReAct agent (bắt buộc search trước, answer sau)."""
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

        # BỎ TRY-EXCEPT TẠM THỜI ĐỂ DEBUG LỖI THỰC TẾ TỪ GEMINI
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
        print(f"\n[DEBUG GEMINI RAW OUTPUT]:\n{text}\n")
        
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
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
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
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids})
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}


if __name__ == '__main__':
    # Kiểm thử nhanh Agent trực tiếp
    retriever = ChromaRetriever()
    controller = GeminiController()
    
    test_question = "Tiêu chuẩn thiết kế kiến trúc hệ thống phân tán là gì?"
    print(f"Đang chạy kiểm thử với câu hỏi: '{test_question}'\n")
    
    result = run_agent(test_question, controller, retriever, top_k=6, budget=4000)
    print("--- KẾT QUẢ AGENT ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))