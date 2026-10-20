"""
RAG Engine orchestrating semantic retrieval, prompt formatting, LLM generation,
and citation grounding for codebase queries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import get_vector_store
from src.llm.base import LLMProvider, get_llm_provider
from src.rag.citations import Citation, extract_citations_from_text, citations_from_chunks
from src.rag.prompt import DEFAULT_SYSTEM_PROMPT, build_rag_prompt

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Represents the complete result of a RAG question-answering query."""
    query: str
    answer: str
    citations: list[Citation]
    sources: list[dict[str, Any]]
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert response to a serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "model": self.model,
            "metadata": self.metadata,
        }

    def format_cited_answer(self) -> str:
        """
        Format the answer with a standardized Citations section.
        """
        if not self.citations:
            return self.answer

        citations_list = "\n".join(f"- `{c.format_citation()}`" for c in self.citations)
        return f"{self.answer}\n\n**Citations:**\n{citations_list}"

    def __str__(self) -> str:
        return self.answer


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for answering natural-language questions
    about codebases with file and line citations.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_provider: LLMProvider | None = None,
        default_collection: str = "codebase",
        system_prompt: str | None = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.llm_provider = llm_provider or get_llm_provider()
        self.default_collection = default_collection
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def answer_question(
        self,
        question: str,
        repo: str | None = None,
        collection_name: str | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Answer a natural-language question grounded in code retrieved from the vector store.
        """
        trimmed_question = question.strip() if question else ""
        model_name = getattr(self.llm_provider, "model", getattr(self.llm_provider, "model_name", "unknown"))

        if not trimmed_question:
            return RAGResponse(
                query=question,
                answer="No question was provided.",
                citations=[],
                sources=[],
                model=model_name,
                metadata={"status": "empty_query"},
            )

        # Determine target collection
        target_collection = collection_name or repo or self.default_collection
        sanitized_col = sanitize_collection_name(target_collection)

        # Step 1: Semantic retrieval
        chunks = self.vector_store.query(
            query_text=trimmed_question,
            top_k=top_k,
            collection_name=sanitized_col,
            where=where,
        )

        # Step 2: Handle empty retrieval
        if not chunks:
            return RAGResponse(
                query=trimmed_question,
                answer=(
                    f"No relevant code context was found in collection '{sanitized_col}' "
                    f"to answer the question: '{trimmed_question}'."
                ),
                citations=[],
                sources=[],
                model=model_name,
                metadata={
                    "collection": sanitized_col,
                    "retrieved_count": 0,
                    "status": "no_context",
                },
            )

        # Step 3: Construct prompt with code context
        user_prompt = build_rag_prompt(trimmed_question, chunks)
        sys_prompt = system_instruction or self.system_prompt

        # Step 4: Generate LLM response
        llm_response = self.llm_provider.generate(
            prompt=user_prompt,
            system_instruction=sys_prompt,
            temperature=temperature,
            **kwargs,
        )

        # Step 5: Extract and reconcile citations
        in_text_citations = extract_citations_from_text(llm_response.content)
        if in_text_citations:
            citations = in_text_citations
        else:
            # Fallback to citations directly from top retrieved chunks
            citations = citations_from_chunks(chunks)

        return RAGResponse(
            query=trimmed_question,
            answer=llm_response.content,
            citations=citations,
            sources=chunks,
            model=llm_response.model,
            metadata={
                "collection": sanitized_col,
                "retrieved_count": len(chunks),
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "llm_metadata": llm_response.metadata,
            },
        )


def ask_codebase(
    question: str,
    repo: str | None = None,
    collection_name: str | None = None,
    vector_store: VectorStore | None = None,
    llm_provider: LLMProvider | None = None,
    top_k: int = 5,
    **kwargs: Any,
) -> RAGResponse:
    """
    Convenience function to ask a question about an indexed codebase.
    """
    engine = RAGEngine(vector_store=vector_store, llm_provider=llm_provider)
    return engine.answer_question(
        question=question,
        repo=repo,
        collection_name=collection_name,
        top_k=top_k,
        **kwargs,
    )
