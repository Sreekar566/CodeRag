"""
RAG Engine orchestrating semantic retrieval, context assembly within token budget,
LLM generation, and citation grounding for codebase queries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.storage.base import VectorStore, sanitize_collection_name
from src.storage.chroma_store import get_vector_store
from src.llm.base import LLMProvider, get_llm_provider
from src.rag.citations import Citation, extract_citations_from_text, citations_from_chunks
from src.rag.context_assembler import ContextAssembler, DEFAULT_MAX_CONTEXT_TOKENS
from src.rag.prompt import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from src.registry.registry import RepoRegistry, get_repo_registry
from src.observability.models import StageTimer, calculate_llm_cost, QueryTelemetry
from src.observability.collector import get_telemetry_collector

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
    about codebases with file and line citations, token-budget enforcement, and re-ranking.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_provider: LLMProvider | None = None,
        registry: RepoRegistry | None = None,
        default_collection: str = "codebase",
        system_prompt: str | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        rerank: bool = True,
        merge_overlapping: bool = True,
        reorder_context: bool = True,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.llm_provider = llm_provider or get_llm_provider()
        self.registry = registry
        self.default_collection = default_collection
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_context_tokens = max_context_tokens
        self.rerank = rerank
        self.merge_overlapping = merge_overlapping
        self.reorder_context = reorder_context

    def _resolve_collection(self, repo_or_col: str) -> tuple[str, str]:
        """
        Resolve a repo name or collection name to (repo_id, collection_name).
        """
        registry = self.registry or get_repo_registry()
        meta = registry.get(repo_or_col)
        if meta is not None:
            return meta.repo_id, meta.collection_name
        return repo_or_col, sanitize_collection_name(repo_or_col)

    def answer_question(
        self,
        question: str,
        repo: str | None = None,
        repos: list[str] | None = None,
        collection_name: str | None = None,
        collections: list[str] | None = None,
        top_k: int = 10,
        max_context_tokens: int | None = None,
        rerank: bool | None = None,
        merge_overlapping: bool | None = None,
        where: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Answer a natural-language question grounded in code retrieved from the vector store.
        Supports single repo/collection or cross-repo queries across multiple repositories.
        Applies token budgeting and re-ranking to assemble optimal context.
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

        # Determine target repositories / collections
        target_specs: list[tuple[str, str]] = []  # list of (repo_label, collection_name)

        if repos:
            for r in repos:
                target_specs.append(self._resolve_collection(r))
        elif collections:
            for c in collections:
                target_specs.append((c, sanitize_collection_name(c)))
        elif repo:
            if isinstance(repo, list):
                for r in repo:
                    target_specs.append(self._resolve_collection(r))
            else:
                target_specs.append(self._resolve_collection(repo))
        elif collection_name:
            target_specs.append((collection_name, sanitize_collection_name(collection_name)))
        else:
            target_specs.append((self.default_collection, sanitize_collection_name(self.default_collection)))

        is_multi_repo = len(target_specs) > 1

        overall_timer = StageTimer().start()

        # Step 1: Semantic retrieval across all targets
        retrieval_timer = StageTimer().start()
        raw_chunks: list[dict[str, Any]] = []
        for repo_label, col_name in target_specs:
            chunks = self.vector_store.query(
                query_text=trimmed_question,
                top_k=top_k,
                collection_name=col_name,
                where=where,
            )
            for c in chunks:
                if is_multi_repo or repo:
                    c["repo"] = repo_label
                    if "metadata" not in c or not isinstance(c["metadata"], dict):
                        c["metadata"] = {}
                    c["metadata"]["repo"] = repo_label
            raw_chunks.extend(chunks)
        retrieval_ms = retrieval_timer.stop()

        # Step 2: Handle empty retrieval
        if not raw_chunks:
            total_duration_ms = overall_timer.stop()
            if is_multi_repo:
                col_display = ", ".join(t[1] for t in target_specs)
                msg = f"No relevant code context was found in collections '{col_display}' to answer the question: '{trimmed_question}'."
                primary_col = "multi"
            else:
                primary_col = target_specs[0][1]
                msg = f"No relevant code context was found in collection '{primary_col}' to answer the question: '{trimmed_question}'."

            telemetry = QueryTelemetry(
                question=trimmed_question,
                target_repos=[t[0] for t in target_specs],
                model=model_name,
                stage_latencies_ms={"retrieval": round(retrieval_ms, 2), "total": round(total_duration_ms, 2)},
                total_duration_ms=total_duration_ms,
                retrieved_count=0,
                selected_count=0,
                citations_count=0,
            )
            get_telemetry_collector().record_query(telemetry)

            return RAGResponse(
                query=trimmed_question,
                answer=msg,
                citations=[],
                sources=[],
                model=model_name,
                metadata={
                    "collection": primary_col,
                    "collections": [t[1] for t in target_specs],
                    "repositories": [t[0] for t in target_specs],
                    "retrieved_count": 0,
                    "status": "no_context",
                    "duration_ms": round(total_duration_ms, 2),
                    "elapsed_seconds": round(total_duration_ms / 1000.0, 3),
                    "stage_durations_ms": {"retrieval": round(retrieval_ms, 2)},
                    "estimated_cost_usd": 0.0,
                    "query_id": telemetry.query_id,
                },
            )

        # Step 3: Context assembly within token budget & re-ranking
        assembly_timer = StageTimer().start()
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        do_rerank = rerank if rerank is not None else self.rerank
        do_merge = merge_overlapping if merge_overlapping is not None else self.merge_overlapping

        assembler = ContextAssembler(
            max_context_tokens=budget,
            rerank=do_rerank,
            merge_overlapping=do_merge,
            reorder_lost_in_the_middle=self.reorder_context,
        )
        assembly_result = assembler.assemble(trimmed_question, raw_chunks)
        selected_chunks = assembly_result.chunks
        formatted_context = assembly_result.formatted_context
        assembly_ms = assembly_timer.stop()

        # Step 4: Construct prompt with code context
        user_prompt = build_rag_prompt(trimmed_question, formatted_context)
        sys_prompt = system_instruction or self.system_prompt

        # Step 5: Generate LLM response
        generation_timer = StageTimer().start()
        llm_response = self.llm_provider.generate(
            prompt=user_prompt,
            system_instruction=sys_prompt,
            temperature=temperature,
            **kwargs,
        )
        generation_ms = generation_timer.stop()

        # Step 6: Extract and reconcile citations
        citations_timer = StageTimer().start()
        in_text_citations = extract_citations_from_text(llm_response.content)
        if in_text_citations:
            citations = in_text_citations
        else:
            # Fallback to citations directly from top assembled chunks
            citations = citations_from_chunks(selected_chunks)
        citations_ms = citations_timer.stop()

        total_duration_ms = overall_timer.stop()

        # Step 7: Cost and telemetry calculation
        estimated_cost_usd = calculate_llm_cost(
            model_name=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )

        stage_latencies = {
            "retrieval": round(retrieval_ms, 2),
            "assembly_rerank": round(assembly_ms, 2),
            "generation": round(generation_ms, 2),
            "citations": round(citations_ms, 2),
        }

        telemetry = QueryTelemetry(
            question=trimmed_question,
            target_repos=[t[0] for t in target_specs],
            model=llm_response.model,
            stage_latencies_ms=stage_latencies,
            total_duration_ms=total_duration_ms,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            context_tokens=assembly_result.total_tokens,
            cost_usd=estimated_cost_usd,
            retrieved_count=len(raw_chunks),
            selected_count=len(selected_chunks),
            citations_count=len(citations),
        )
        get_telemetry_collector().record_query(telemetry)

        primary_col = target_specs[0][1] if len(target_specs) == 1 else "multi"

        return RAGResponse(
            query=trimmed_question,
            answer=llm_response.content,
            citations=citations,
            sources=selected_chunks,
            model=llm_response.model,
            metadata={
                "collection": primary_col,
                "collections": [t[1] for t in target_specs],
                "repositories": [t[0] for t in target_specs],
                "multi_repo": is_multi_repo,
                "retrieved_count": len(raw_chunks),
                "selected_count": len(selected_chunks),
                "context_tokens": assembly_result.total_tokens,
                "max_context_tokens": budget,
                "truncated_chunks_count": assembly_result.truncated_count,
                "dropped_chunks_count": assembly_result.dropped_count,
                "reordered": assembly_result.reordered,
                "reranked": do_rerank,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "duration_ms": round(total_duration_ms, 2),
                "elapsed_seconds": round(total_duration_ms / 1000.0, 3),
                "stage_durations_ms": stage_latencies,
                "estimated_cost_usd": estimated_cost_usd,
                "query_id": telemetry.query_id,
                "llm_metadata": llm_response.metadata,
            },
        )

    def answer_multi_repo(
        self,
        question: str,
        repos: list[str],
        top_k_per_repo: int = 5,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        Query across multiple repositories, rank best chunks across all codebases,
        and generate a cross-repo grounded answer.
        """
        return self.answer_question(
            question=question,
            repos=repos,
            top_k=top_k_per_repo,
            **kwargs,
        )


def ask_codebase(
    question: str,
    repo: str | None = None,
    repos: list[str] | None = None,
    collection_name: str | None = None,
    collections: list[str] | None = None,
    vector_store: VectorStore | None = None,
    llm_provider: LLMProvider | None = None,
    top_k: int = 10,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    rerank: bool = True,
    merge_overlapping: bool = True,
    registry: RepoRegistry | None = None,
    **kwargs: Any,
) -> RAGResponse:
    """
    Convenience function to ask a question about an indexed codebase with context budgeting.
    Supports single repository or multi-repo questions.
    """
    engine = RAGEngine(
        vector_store=vector_store,
        llm_provider=llm_provider,
        registry=registry,
        max_context_tokens=max_context_tokens,
        rerank=rerank,
        merge_overlapping=merge_overlapping,
    )
    return engine.answer_question(
        question=question,
        repo=repo,
        repos=repos,
        collection_name=collection_name,
        collections=collections,
        top_k=top_k,
        **kwargs,
    )
