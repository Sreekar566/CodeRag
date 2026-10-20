"""
Unit tests for context assembler, token budgeting, and lost-in-the-middle re-ordering.
"""
import pytest

from src.rag.context_assembler import (
    ContextAssembler,
    reorder_for_attention,
    assemble_context,
)
from src.rag.tokenizer import count_tokens


class TestContextAssembler:
    def test_assemble_empty_chunks(self):
        assembler = ContextAssembler(max_context_tokens=1000)
        res = assembler.assemble("Where is main?", [])
        assert res.selected_count == 0
        assert res.retrieved_count == 0
        assert "No relevant code context found." in res.formatted_context

    def test_assemble_fits_within_budget(self):
        chunks = [
            {
                "path": "app/routes.py",
                "start_line": 1,
                "end_line": 10,
                "language": "python",
                "text": "def home():\n    return 'home'",
                "similarity": 0.9,
            },
            {
                "path": "app/models.py",
                "start_line": 1,
                "end_line": 10,
                "language": "python",
                "text": "class User:\n    id = 1",
                "similarity": 0.85,
            },
        ]
        assembler = ContextAssembler(max_context_tokens=1000)
        res = assembler.assemble("What routes and models exist?", chunks)
        assert res.selected_count == 2
        assert res.dropped_count == 0
        assert res.total_tokens <= 1000
        assert "[Source 1]" in res.formatted_context
        assert "[Source 2]" in res.formatted_context

    def test_assemble_drops_chunks_when_exceeding_budget(self):
        # Create 10 chunks that each take ~30-40 tokens
        chunks = [
            {
                "path": f"app/module_{i}.py",
                "start_line": 1,
                "end_line": 10,
                "language": "python",
                "text": f"def func_{i}():\n    x = {i} * 100\n    return x",
                "similarity": 0.9 - (i * 0.05),
            }
            for i in range(10)
        ]
        # Restrict budget so only ~2 chunks fit
        assembler = ContextAssembler(
            max_context_tokens=100,
            allow_truncation=False,
            rerank=False,
            merge_overlapping=False,
            reorder_lost_in_the_middle=False,
        )
        res = assembler.assemble("test query", chunks)
        assert res.selected_count < 10
        assert res.dropped_count > 0
        assert res.total_tokens <= 100

    def test_assemble_truncates_last_chunk_to_maximize_budget(self):
        small_chunk = {
            "path": "app/short.py",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "text": "x = 1\ny = 2",
            "similarity": 0.9,
        }
        large_chunk = {
            "path": "app/long.py",
            "start_line": 1,
            "end_line": 100,
            "language": "python",
            "text": "\n".join([f"var_{i} = 'line content {i}'" for i in range(100)]),
            "similarity": 0.8,
        }
        assembler = ContextAssembler(
            max_context_tokens=150,
            allow_truncation=True,
            rerank=False,
            merge_overlapping=False,
            reorder_lost_in_the_middle=False,
        )
        res = assembler.assemble("test query", [small_chunk, large_chunk])
        assert res.selected_count == 2
        assert res.truncated_count == 1
        assert res.chunks[1].get("truncated") is True
        assert "[Truncated]" in res.formatted_context
        assert res.total_tokens <= 170

    def test_reorder_for_attention(self):
        # Test lost-in-the-middle alternating placement
        chunks = [{"id": i, "score": 10 - i} for i in range(5)]
        # Order is [c0, c1, c2, c3, c4]
        reordered = reorder_for_attention(chunks)
        ids = [c["id"] for c in reordered]
        # c0 at start, c1 at end, c2 next, c3 before end, c4 in middle
        assert ids == [0, 2, 4, 3, 1]

    def test_assemble_context_convenience_function(self):
        chunks = [
            {
                "path": "main.py",
                "start_line": 1,
                "end_line": 5,
                "language": "python",
                "text": "print('hello')",
                "similarity": 0.9,
            }
        ]
        res = assemble_context("test", chunks, max_tokens=500)
        assert res.selected_count == 1
        assert res.total_tokens > 0
