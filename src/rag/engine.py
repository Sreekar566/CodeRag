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
from src.rag.context_assembler import (
    ContextAssembler,
    ContextAssemblyResult,
    DEFAULT_MAX_CONTEXT_TOKENS,
)
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

    def _resolve_target_specs(
        self,
        repo: str | list[str] | None = None,
        repos: list[str] | None = None,
        collection_name: str | None = None,
        collections: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """
        Determine target repositories and collections to query.
        Returns a deduplicated list of (repo_label, collection_name) tuples.
        """
        target_specs: list[tuple[str, str]] = []

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

        # Deduplicate while preserving order
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for spec in target_specs:
            if spec not in seen:
                seen.add(spec)
                deduped.append(spec)
        return deduped

    def _retrieve_chunks(
        self,
        query: str,
        target_specs: list[tuple[str, str]],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        tag_repo: bool = False,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Query vector storage across all target collections and annotate chunks with repo metadata.
        Returns (raw_chunks, retrieval_latency_ms).
        """
        timer = StageTimer().start()
        raw_chunks: list[dict[str, Any]] = []

        for repo_label, col_name in target_specs:
            chunks = self.vector_store.query(
                query_text=query,
                top_k=top_k,
                collection_name=col_name,
                where=where,
            )
            for c in chunks:
                if tag_repo:
                    c["repo"] = repo_label
                    if "metadata" not in c or not isinstance(c["metadata"], dict):
                        c["metadata"] = {}
                    c["metadata"]["repo"] = repo_label
            raw_chunks.extend(chunks)

        retrieval_ms = timer.stop()
        return raw_chunks, retrieval_ms

    def _build_empty_response(
        self,
        query: str,
        target_specs: list[tuple[str, str]],
        model_name: str,
        retrieval_ms: float,
        overall_timer: StageTimer,
    ) -> RAGResponse:
        """
        Construct a standardized empty-context response and log telemetry when no chunks are found.
        """
        total_duration_ms = overall_timer.stop()
        is_multi_repo = len(target_specs) > 1

        if is_multi_repo:
            col_display = ", ".join(t[1] for t in target_specs)
            msg = f"No relevant code context was found in collections '{col_display}' to answer the question: '{query}'."
            primary_col = "multi"
        else:
            primary_col = target_specs[0][1]
            msg = f"No relevant code context was found in collection '{primary_col}' to answer the question: '{query}'."

        stage_latencies = {
            "retrieval": round(retrieval_ms, 2),
            "total": round(total_duration_ms, 2),
        }

        telemetry = QueryTelemetry(
            question=query,
            target_repos=[t[0] for t in target_specs],
            model=model_name,
            stage_latencies_ms=stage_latencies,
            total_duration_ms=total_duration_ms,
            retrieved_count=0,
            selected_count=0,
            citations_count=0,
        )
        get_telemetry_collector().record_query(telemetry)

        return RAGResponse(
            query=query,
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

    def _assemble_context(
        self,
        query: str,
        raw_chunks: list[dict[str, Any]],
        budget: int,
        rerank: bool,
        merge_overlapping: bool,
    ) -> tuple[ContextAssemblyResult, float]:
        """
        Assemble retrieved chunks to fit within the token budget, applying reranking,
        merging, and attention-based reordering.
        Returns (assembly_result, assembly_latency_ms).
        """
        timer = StageTimer().start()
        assembler = ContextAssembler(
            max_context_tokens=budget,
            rerank=rerank,
            merge_overlapping=merge_overlapping,
            reorder_lost_in_the_middle=self.reorder_context,
        )
        assembly_result = assembler.assemble(query, raw_chunks)
        assembly_ms = timer.stop()
        return assembly_result, assembly_ms

    def _generate_answer(
        self,
        query: str,
        formatted_context: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        """
        Format the prompt with code context and invoke the LLM provider.
        Returns (llm_response, generation_latency_ms).
        """
        timer = StageTimer().start()
        user_prompt = build_rag_prompt(query, formatted_context)
        sys_prompt = system_instruction or self.system_prompt
        llm_response = self.llm_provider.generate(
            prompt=user_prompt,
            system_instruction=sys_prompt,
            temperature=temperature,
            **kwargs,
        )
        generation_ms = timer.stop()
        return llm_response, generation_ms

    def _resolve_citations(
        self,
        answer_text: str,
        selected_chunks: list[dict[str, Any]],
    ) -> tuple[list[Citation], float]:
        """
        Extract citations from generated text, falling back to top source chunks if none found.
        Enriches in-text citations with repository, language, and snippet metadata from selected chunks.
        Returns (citations, citation_resolution_latency_ms).
        """
        timer = StageTimer().start()
        in_text_citations = extract_citations_from_text(answer_text)
        if in_text_citations:
            citations = in_text_citations
            if selected_chunks:
                for cit in citations:
                    for chunk in selected_chunks:
                        chunk_path = chunk.get("path") or (chunk.get("metadata", {}) or {}).get("path")
                        if not chunk_path:
                            continue
                        chunk_repo = chunk.get("repo") or (chunk.get("metadata", {}) or {}).get("repo")

                        norm_cit_path = cit.path.strip().replace("\\", "/")
                        while norm_cit_path.startswith("./"):
                            norm_cit_path = norm_cit_path[2:]
                        norm_cit_path = norm_cit_path.lstrip("/")

                        norm_chunk_path = str(chunk_path).strip().replace("\\", "/")
                        while norm_chunk_path.startswith("./"):
                            norm_chunk_path = norm_chunk_path[2:]
                        norm_chunk_path = norm_chunk_path.lstrip("/")

                        path_matches = (
                            norm_cit_path.lower() == norm_chunk_path.lower()
                            or norm_cit_path.lower().endswith("/" + norm_chunk_path.lower())
                            or norm_chunk_path.lower().endswith("/" + norm_cit_path.lower())
                        )
                        if path_matches:
                            if cit.repo and chunk_repo and cit.repo != chunk_repo:
                                continue
                            if not cit.repo and chunk_repo:
                                cit.repo = chunk_repo
                            if not cit.language:
                                cit.language = chunk.get("language") or (chunk.get("metadata", {}) or {}).get("language")
                            if not cit.snippet:
                                cit.snippet = chunk.get("text")
                            break
        else:
            citations = citations_from_chunks(selected_chunks)
        citations_ms = timer.stop()
        return citations, citations_ms

    def _record_telemetry_and_build_response(
        self,
        query: str,
        target_specs: list[tuple[str, str]],
        raw_chunks: list[dict[str, Any]],
        assembly_result: ContextAssemblyResult,
        llm_response: Any,
        citations: list[Citation],
        stage_latencies: dict[str, float],
        total_duration_ms: float,
        rerank: bool,
        budget: int,
    ) -> RAGResponse:
        """
        Calculate token costs, log telemetry, and package the final RAGResponse with execution metadata.
        """
        is_multi_repo = len(target_specs) > 1
        estimated_cost_usd = calculate_llm_cost(
            model_name=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )

        telemetry = QueryTelemetry(
            question=query,
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
            selected_count=len(assembly_result.chunks),
            citations_count=len(citations),
        )
        get_telemetry_collector().record_query(telemetry)

        primary_col = target_specs[0][1] if len(target_specs) == 1 else "multi"

        return RAGResponse(
            query=query,
            answer=llm_response.content,
            citations=citations,
            sources=assembly_result.chunks,
            model=llm_response.model,
            metadata={
                "collection": primary_col,
                "collections": [t[1] for t in target_specs],
                "repositories": [t[0] for t in target_specs],
                "multi_repo": is_multi_repo,
                "retrieved_count": len(raw_chunks),
                "selected_count": len(assembly_result.chunks),
                "context_tokens": assembly_result.total_tokens,
                "max_context_tokens": budget,
                "truncated_chunks_count": assembly_result.truncated_count,
                "dropped_chunks_count": assembly_result.dropped_count,
                "reordered": assembly_result.reordered,
                "reranked": rerank,
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

        overall_timer = StageTimer().start()

        # Step 0: Resolve target repositories / collections
        target_specs = self._resolve_target_specs(
            repo=repo,
            repos=repos,
            collection_name=collection_name,
            collections=collections,
        )
        tag_repo = len(target_specs) > 1 or repo is not None or repos is not None

        # Step 1: Semantic retrieval across all targets
        raw_chunks, retrieval_ms = self._retrieve_chunks(
            query=trimmed_question,
            target_specs=target_specs,
            top_k=top_k,
            where=where,
            tag_repo=tag_repo,
        )

        # Step 2: Handle empty retrieval
        if not raw_chunks:
            return self._build_empty_response(
                query=trimmed_question,
                target_specs=target_specs,
                model_name=model_name,
                retrieval_ms=retrieval_ms,
                overall_timer=overall_timer,
            )

        # Step 3: Context assembly within token budget & re-ranking
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        do_rerank = rerank if rerank is not None else self.rerank
        do_merge = merge_overlapping if merge_overlapping is not None else self.merge_overlapping

        assembly_result, assembly_ms = self._assemble_context(
            query=trimmed_question,
            raw_chunks=raw_chunks,
            budget=budget,
            rerank=do_rerank,
            merge_overlapping=do_merge,
        )

        # Step 4 & 5: Construct prompt and generate LLM response
        llm_response, generation_ms = self._generate_answer(
            query=trimmed_question,
            formatted_context=assembly_result.formatted_context,
            system_instruction=system_instruction,
            temperature=temperature,
            **kwargs,
        )

        # Step 6: Extract and reconcile citations
        citations, citations_ms = self._resolve_citations(
            answer_text=llm_response.content,
            selected_chunks=assembly_result.chunks,
        )

        total_duration_ms = overall_timer.stop()

        stage_latencies = {
            "retrieval": round(retrieval_ms, 2),
            "assembly_rerank": round(assembly_ms, 2),
            "generation": round(generation_ms, 2),
            "citations": round(citations_ms, 2),
        }

        # Step 7: Record telemetry and assemble final response
        return self._record_telemetry_and_build_response(
            query=trimmed_question,
            target_specs=target_specs,
            raw_chunks=raw_chunks,
            assembly_result=assembly_result,
            llm_response=llm_response,
            citations=citations,
            stage_latencies=stage_latencies,
            total_duration_ms=total_duration_ms,
            rerank=do_rerank,
            budget=budget,
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
