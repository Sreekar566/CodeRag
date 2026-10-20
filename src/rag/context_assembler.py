"""
Context assembly and token budget management for codebase RAG.
Optimizes context window utilization, applies token limits, merges overlapping chunks,
and places high-priority snippets strategically to avoid lost-in-the-middle degradation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.rag.prompt import format_code_context
from src.rag.reranker import rerank_chunks
from src.rag.tokenizer import count_tokens, estimate_chunk_tokens, truncate_text_to_tokens

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 2048


@dataclass
class ContextAssemblyResult:
    """Represents the outcome of assembling context chunks within a token budget."""
    chunks: list[dict[str, Any]]
    formatted_context: str
    total_tokens: int
    max_budget: int
    retrieved_count: int
    selected_count: int
    dropped_count: int
    truncated_count: int
    reordered: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "max_budget": self.max_budget,
            "retrieved_count": self.retrieved_count,
            "selected_count": self.selected_count,
            "dropped_count": self.dropped_count,
            "truncated_count": self.truncated_count,
            "reordered": self.reordered,
            "metadata": self.metadata,
        }


def reorder_for_attention(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reorder chunks to mitigate LLM 'lost in the middle' attention decay.
    Places the highest-scoring chunks at the beginning and the very end of the context,
    with lower-scoring chunks placed in the center.
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict[str, Any] | None] = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1

    return [c for c in reordered if c is not None]


class ContextAssembler:
    """
    Assembles code context chunks to fit within a strict token budget.
    Handles re-ranking, deduplication/merging, token estimation, truncation,
    and attention-optimal reordering.
    """

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        rerank: bool = True,
        merge_overlapping: bool = True,
        reorder_lost_in_the_middle: bool = True,
        allow_truncation: bool = True,
        min_chunk_tokens: int = 50,
    ):
        self.max_context_tokens = max_context_tokens
        self.rerank = rerank
        self.merge_overlapping = merge_overlapping
        self.reorder_lost_in_the_middle = reorder_lost_in_the_middle
        self.allow_truncation = allow_truncation
        self.min_chunk_tokens = min_chunk_tokens

    def assemble(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> ContextAssemblyResult:
        """
        Assemble retrieved chunks into an optimized context block conforming to the token limit.
        """
        budget = max_tokens if max_tokens is not None else self.max_context_tokens
        retrieved_count = len(retrieved_chunks)

        if not retrieved_chunks:
            return ContextAssemblyResult(
                chunks=[],
                formatted_context="No relevant code context found.",
                total_tokens=count_tokens("No relevant code context found."),
                max_budget=budget,
                retrieved_count=0,
                selected_count=0,
                dropped_count=0,
                truncated_count=0,
                reordered=False,
            )

        # Step 1: Re-ranking and chunk merging
        if self.rerank:
            candidates = rerank_chunks(
                query=query,
                chunks=retrieved_chunks,
                merge_overlapping=self.merge_overlapping,
            )
        elif self.merge_overlapping:
            from src.rag.reranker import merge_overlapping_chunks
            candidates = merge_overlapping_chunks(retrieved_chunks)
        else:
            candidates = [c.copy() for c in retrieved_chunks]

        # Step 2: Greedily select chunks within token budget
        selected: list[dict[str, Any]] = []
        accumulated_tokens = 0
        truncated_count = 0

        for chunk in candidates:
            chunk_tokens = estimate_chunk_tokens(chunk)

            if accumulated_tokens + chunk_tokens <= budget:
                selected.append(chunk)
                accumulated_tokens += chunk_tokens
            else:
                remaining_budget = budget - accumulated_tokens
                if self.allow_truncation and remaining_budget >= self.min_chunk_tokens:
                    # Truncate text of this chunk to fit remainder
                    header_cost = estimate_chunk_tokens({**chunk, "text": ""})
                    available_for_text = max(10, remaining_budget - header_cost)
                    truncated_text, _ = truncate_text_to_tokens(
                        chunk.get("text", ""),
                        max_tokens=available_for_text,
                        preserve_lines=True,
                    )

                    truncated_chunk = chunk.copy()
                    truncated_chunk["text"] = truncated_text
                    truncated_chunk["truncated"] = True
                    truncated_chunk_tokens = estimate_chunk_tokens(truncated_chunk)

                    selected.append(truncated_chunk)
                    accumulated_tokens += truncated_chunk_tokens
                    truncated_count += 1
                # Stop selecting further chunks as budget is exhausted
                break

        dropped_count = retrieved_count - len(selected)

        # Step 3: Attention optimization (Lost in the middle)
        was_reordered = False
        if self.reorder_lost_in_the_middle and len(selected) > 2:
            final_chunks = reorder_for_attention(selected)
            was_reordered = True
        else:
            final_chunks = selected

        # Step 4: Final formatting
        formatted = format_code_context(final_chunks)
        final_token_count = count_tokens(formatted)

        return ContextAssemblyResult(
            chunks=final_chunks,
            formatted_context=formatted,
            total_tokens=final_token_count,
            max_budget=budget,
            retrieved_count=retrieved_count,
            selected_count=len(final_chunks),
            dropped_count=dropped_count,
            truncated_count=truncated_count,
            reordered=was_reordered,
            metadata={
                "accumulated_estimate": accumulated_tokens,
                "reranked": self.rerank,
                "merged": self.merge_overlapping,
            },
        )


def assemble_context(
    query: str,
    chunks: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    **kwargs: Any,
) -> ContextAssemblyResult:
    """
    Convenience function to assemble and budget context chunks.
    """
    assembler = ContextAssembler(max_context_tokens=max_tokens, **kwargs)
    return assembler.assemble(query, chunks)
