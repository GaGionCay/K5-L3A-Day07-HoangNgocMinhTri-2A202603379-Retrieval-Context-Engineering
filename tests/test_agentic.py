"""Offline behavioral checks only; fake adapters do not measure model quality."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starter.agentic import GeminiController, run_agent

CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày.', 'source_id': 'doc',
         'source': 'doc.pdf', 'page': 1, 'score': 0.8, 'metadata': {}}

class Controller:
    def __init__(self, actions): self.actions = iter(actions)
    def decide(self, question, history): return next(self.actions)

class Retriever:
    def __init__(self, results): self.results = results; self.calls = []
    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k)); return self.results

class AgentTests(unittest.TestCase):
    def test_gemini_controller_normalizes_citations_to_context_ids(self):
        response = SimpleNamespace(text='{"action":"answer","answer":"Refunds are available within 7 days [doc_01_chunk_0]. The refund request must include a receipt [doc_01_chunk_1].","citations":["[doc_01_chunk_0]","[doc_01_chunk_1]"]}')
        generate_content = Mock(return_value=response)
        controller = GeminiController.__new__(GeminiController)
        controller.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        controller.model_name = 'test-model'

        chunks = [
            {**CHUNK, 'id': 'doc_01_chunk_0'},
            {**CHUNK, 'id': 'doc_01_chunk_1', 'text': 'Cần có hóa đơn để yêu cầu hoàn tiền.'}
        ]
        result = run_agent('When can I get a refund?', controller, Retriever(chunks))

        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['doc_01_chunk_0', 'doc_01_chunk_1'])
        self.assertIn('[doc_01_chunk_0]', result['answer'])
        self.assertIn('[doc_01_chunk_1]', result['answer'])
        instruction = generate_content.call_args.kwargs['config']['system_instruction']
        self.assertIn('doc_01_chunk_0', instruction)
        self.assertIn('doc_01_chunk_1', instruction)

    def test_gemini_controller_rejects_citations_outside_context(self):
        response = SimpleNamespace(text='{"action":"answer","answer":"Refunds are available [not_in_context].","citations":["not_in_context"]}')
        controller = GeminiController.__new__(GeminiController)
        controller.client = SimpleNamespace(models=SimpleNamespace(
            generate_content=Mock(return_value=response)
        ))
        controller.model_name = 'test-model'

        result = run_agent('When can I get a refund?', controller, Retriever([CHUNK]))

        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['citations'], [])

    def test_agent_searches_and_cites_actual_context(self):
        c = Controller([{'action': 'search', 'query': 'hoàn tiền'},
                        {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK]); result = run_agent('Hoàn tiền?', c, r, filters={'year': 2025})
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['c1'])
        self.assertEqual(r.calls, [('hoàn tiền', {'year': 2025}, 3)])
        self.assertEqual(result['events'][0]['context_ids'], ['c1'])

    def test_invalid_citation_and_empty_context_abstain(self):
        for rows, cid in (([CHUNK], 'not-present'), ([], 'c1')):
            c = Controller([{'action':'search','query':'q'}, {'action':'answer','answer':f'7 ngày [{cid}]','citations':[cid]}])
            self.assertEqual(run_agent('q', c, Retriever(rows))['status'], 'abstain')

    def test_search_limit_is_enforced(self):
        c = Controller([{'action':'search','query':'one'}, {'action':'search','query':'two'}, {'action':'search','query':'three'}])
        r = Retriever([CHUNK]); result = run_agent('q', c, r, max_searches=2)
        self.assertEqual(result['status'], 'abstain'); self.assertEqual(len(r.calls), 2)

    def test_context_budget_across_multiple_searches(self):
        other = {**CHUNK, 'id': 'c2', 'text': 'Một đoạn bằng chứng khác.'}
        class SequentialRetriever:
            def __init__(self): self.i = 0
            def search(self, query, *, filters, top_k):
                self.i += 1
                return [CHUNK] if self.i == 1 else [other]
        class InspectController:
            def __init__(self): self.i = 0
            def decide(self, question, history):
                self.i += 1
                if history:
                    assert len(history) == 1 and len(history[0]['context']) <= 90
                if self.i <= 2: return {'action': 'search', 'query': str(self.i)}
                return {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}
        result = run_agent('q', InspectController(), SequentialRetriever(), budget=90)
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(all(len(e['context']) <= 90 for e in result['events'] if e['event'] == 'search'))

    def test_configurable_retrieval_allows_more_than_two_calls(self):
        c = Controller([{'action': 'search', 'query': str(i)} for i in range(3)] +
                       [{'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK])
        result = run_agent('q', c, r, max_searches=3, top_k=7, budget=1200)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(r.calls), 3)
        self.assertTrue(all(call[2] == 7 for call in r.calls))

    def test_tool_errors_are_not_hidden(self):
        class Broken:
            def search(self, query, *, filters, top_k): raise ConnectionError('database unavailable')
        with self.assertRaises(ConnectionError):
            run_agent('q', Controller([{'action':'search','query':'q'}]), Broken())

    def test_gemini_controller_retry_backoff(self):
        calls = []
        def flaky_generate(*args, **kwargs):
            calls.append(len(calls))
            if len(calls) < 3:
                raise Exception("429 RESOURCE_EXHAUSTED quota exceeded")
            return SimpleNamespace(text='{"action":"answer","answer":"OK [c1]","citations":["c1"]}')

        controller = GeminiController.__new__(GeminiController)
        controller.client = SimpleNamespace(models=SimpleNamespace(generate_content=flaky_generate))
        controller.model_name = 'test-model'
        controller.max_retries = 3
        controller.base_delay = 0.01  # Fast test delay

        history = [{'tool': 'search', 'context': '[c1 | test.txt | page 1]\nOK', 'context_ids': ['c1']}]
        res = controller.decide("What?", history)
        self.assertEqual(res['action'], 'answer')
        self.assertEqual(len(calls), 3)

if __name__ == '__main__': unittest.main()
