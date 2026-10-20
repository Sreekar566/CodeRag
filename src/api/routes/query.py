"""
Natural language codebase query endpoint with RAG context assembly and code citations.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_rag_engine
from src.api.models import CitationModel, QueryRequest, QueryResponse, SourceChunkModel
from src.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask natural language questions about indexed codebases",
)
def query_codebase(
    request: QueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> QueryResponse:
    """
    Search vector collections for relevant code snippets across one or more repositories,
    assemble token-budgeted prompt context, and generate grounded answers with line citations.
    """
    trimmed = request.question.strip()
    if not trimmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        rag_response = rag_engine.answer_question(
            question=trimmed,
            repo=request.repo,
            repos=request.repos,
            collection_name=request.collection_name,
            collections=request.collections,
            top_k=request.top_k,
            max_context_tokens=request.max_context_tokens,
            rerank=request.rerank,
            merge_overlapping=request.merge_overlapping,
            where=request.where,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"Error executing codebase query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {e}",
        )

    citations = [
        CitationModel(
            path=c.path,
            start_line=c.start_line,
            end_line=c.end_line,
            repo=c.repo,
            citation_str=c.format_citation(),
        )
        for c in rag_response.citations
    ]

    sources = [
        SourceChunkModel(
            id=str(s.get("id", "")),
            text=str(s.get("text", "")),
            path=s.get("path"),
            start_line=s.get("start_line"),
            end_line=s.get("end_line"),
            language=s.get("language"),
            similarity=s.get("similarity"),
            distance=s.get("distance"),
            repo=s.get("repo"),
            metadata=s.get("metadata", {}),
        )
        for s in rag_response.sources
    ]

    return QueryResponse(
        query=rag_response.query,
        answer=rag_response.answer,
        formatted_answer=rag_response.format_cited_answer(),
        citations=citations,
        sources=sources,
        model=rag_response.model,
        metadata=rag_response.metadata,
    )
