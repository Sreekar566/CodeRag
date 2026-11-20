"""
Developer Command-Line Interface (CLI) for the AI Codebase Intelligence Platform.
Enables terminal-based codebase operations, vector storage inspection,
incremental repository indexing, natural language queries, and Git change preview.

Usage:
    python -m src.cli status
    python -m src.cli repos
    python -m src.cli query "Where is authentication handled?" --repo miguelgrinberg/microblog
    python -m src.cli index owner/repo --branch main
    python -m src.cli changes owner/repo
    python -m src.cli delete owner/repo --yes
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


def parse_repo_arg(repo_arg: str) -> tuple[str, str]:
    """Parse 'owner/repo' string into (owner, repo) tuple."""
    parts = repo_arg.strip().split("/", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Invalid repository format '{repo_arg}'. Expected 'owner/repo' (e.g. 'pallets/flask')."
        )
    return parts[0].strip(), parts[1].strip()


def handle_status(args: argparse.Namespace) -> int:
    """Inspect and display system health, storage statistics, and LLM configuration."""
    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            health = client.health()
            collections = client.list_collections()
            repos = client.list_repos()
            status_data = {
                "status": health.get("status", "ok"),
                "app_name": "AI Codebase Intelligence Platform (API)",
                "version": health.get("version", "unknown"),
                "api_url": args.api_url,
                "vector_store": health.get("vector_store_type", "unknown"),
                "collections": [c.get("name", "") for c in collections],
                "collections_count": len(collections),
                "registered_repos_count": len(repos),
                "llm_provider": health.get("llm_provider", "unknown"),
                "llm_model": health.get("llm_model", "unknown"),
            }
        except ApiClientError as exc:
            print(f"Error connecting to API server at {args.api_url}: {exc}", file=sys.stderr)
            return 1
    else:
        from src.storage.chroma_store import get_vector_store
        from src.registry.registry import get_repo_registry
        from src.llm.base import get_llm_provider

        settings = get_settings()
        storage = get_vector_store()
        registry = get_repo_registry()

        try:
            llm = get_llm_provider(getattr(args, "llm", None))
            llm_class = llm.__class__.__name__
            llm_model = getattr(llm, "model", getattr(llm, "model_name", "unknown"))
        except Exception:
            llm_class = "unknown"
            llm_model = "unknown"

        collections = storage.list_collections()
        repos = registry.list_repos()
        status_data = {
            "status": "ok",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "vector_store": storage.__class__.__name__,
            "collections": collections,
            "collections_count": len(collections),
            "chroma_persist_directory": settings.chroma_persist_directory,
            "registered_repos_count": len(repos),
            "registry_path": settings.registry_path,
            "llm_provider": llm_class,
            "llm_model": str(llm_model),
        }

    if args.json:
        print(json.dumps(status_data, indent=2))
        return 0

    print("AI Codebase Intelligence Platform - Status")
    print("=" * 45)
    print(f"Status:               {status_data['status'].upper()}")
    print(f"Version:              {status_data['version']}")
    print(f"Vector Store:         {status_data['vector_store']} ({status_data['collections_count']} collections)")
    if status_data["collections"]:
        col_list = ", ".join(status_data["collections"][:5])
        if len(status_data["collections"]) > 5:
            col_list += f" (+{len(status_data['collections']) - 5} more)"
        print(f"Collections:          {col_list}")
    if "chroma_persist_directory" in status_data:
        print(f"Persistence Dir:      {status_data['chroma_persist_directory']}")
    print(f"Registered Repos:     {status_data['registered_repos_count']}")
    if "registry_path" in status_data:
        print(f"Registry Path:        {status_data['registry_path']}")
    print(f"LLM Provider:         {status_data['llm_provider']} ({status_data['llm_model']})")
    if "api_url" in status_data:
        print(f"API Endpoint:         {status_data['api_url']}")
    return 0


def handle_repos(args: argparse.Namespace) -> int:
    """List all registered and indexed codebases."""
    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            repo_dicts = client.list_repos()
        except ApiClientError as exc:
            print(f"Error querying API at {args.api_url}: {exc}", file=sys.stderr)
            return 1
    else:
        from src.registry.registry import get_repo_registry

        registry = get_repo_registry()
        repos = registry.list_repos()
        repo_dicts = [r.to_dict() for r in repos]

    if args.json:
        print(json.dumps(repo_dicts, indent=2))
        return 0

    if not repo_dicts:
        print("No repositories currently registered or indexed.")
        print("Use 'python -m src.cli index <owner/repo>' to index a codebase.")
        return 0

    print("Registered Codebase Repositories")
    print("=" * 95)
    header = f"{'Repository':<32} {'Branch':<10} {'Commit':<10} {'Files':<8} {'Chunks':<8} {'Status':<8} {'Indexed At'}"
    print(header)
    print("-" * 95)
    for r in repo_dicts:
        repo_name = f"{r.get('owner', '')}/{r.get('repo', '')}"
        branch = r.get("branch", "main") or "main"
        commit = (r.get("commit_sha") or "N/A")[:7]
        files = str(r.get("file_count", 0))
        chunks = str(r.get("chunk_count", 0))
        stat = r.get("status", "unknown")
        indexed_at = (r.get("indexed_at") or "N/A")[:19]
        print(f"{repo_name:<32} {branch:<10} {commit:<10} {files:<8} {chunks:<8} {stat:<8} {indexed_at}")
    print("-" * 95)
    print(f"Total: {len(repo_dicts)} repositories")
    return 0


def handle_query(args: argparse.Namespace) -> int:
    """Query indexed codebases with grounded file & line citations."""
    target_repos = []
    if args.repos:
        target_repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    elif args.repo:
        target_repos = [args.repo.strip()]

    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            res = client.query(
                question=args.question,
                repo=args.repo if not args.repos else None,
                repos=target_repos if len(target_repos) > 1 else None,
                collection_name=args.collection,
                top_k=args.top_k,
                rerank=not args.no_rerank,
                max_context_tokens=args.max_tokens,
            )
        except ApiClientError as exc:
            print(f"Error querying API at {args.api_url}: {exc}", file=sys.stderr)
            return 1
        res_dict = res
    else:
        from src.rag.engine import RAGEngine
        from src.storage.chroma_store import get_vector_store
        from src.registry.registry import get_repo_registry
        from src.llm.base import get_llm_provider

        storage = get_vector_store()
        registry = get_repo_registry()
        llm = get_llm_provider(args.llm)

        engine = RAGEngine(
            vector_store=storage,
            llm_provider=llm,
            registry=registry,
            max_context_tokens=args.max_tokens,
            rerank=not args.no_rerank,
        )
        rag_resp = engine.answer_question(
            question=args.question,
            repo=args.repo if not args.repos else None,
            repos=target_repos if len(target_repos) > 1 else None,
            collection_name=args.collection,
            top_k=args.top_k,
            max_context_tokens=args.max_tokens,
            rerank=not args.no_rerank,
        )
        res_dict = rag_resp.to_dict()

    if args.json:
        print(json.dumps(res_dict, indent=2))
        return 0

    print("AI Codebase Intelligence - Query Result")
    print("=" * 80)
    print(f"Question: {args.question}")
    meta = res_dict.get("metadata", {})
    model = res_dict.get("model", "unknown")
    dur_ms = meta.get("duration_ms", 0.0)
    cost = meta.get("estimated_cost_usd", 0.0)
    tokens = meta.get("total_tokens")
    prompt_tokens = meta.get("prompt_tokens")
    comp_tokens = meta.get("completion_tokens")
    stages = meta.get("stage_durations_ms", {})

    stage_str = (
        " | ".join(f"{k.capitalize()}: {v}ms" for k, v in stages.items())
        if stages
        else f"{dur_ms}ms"
    )
    token_str = f"Prompt: {prompt_tokens or 0} | Completion: {comp_tokens or 0}"
    if tokens is not None:
        token_str += f" | Total: {tokens}"
    token_str += f" | Cost: ${cost:.6f}"

    print(f"Model:    {model}")
    print(f"Timing:   {dur_ms}ms ({stage_str})")
    print(f"Tokens:   {token_str}")
    print("-" * 80)
    print("Answer:")
    print(res_dict.get("answer", ""))
    print("-" * 80)

    citations = res_dict.get("citations", [])
    if citations:
        print("Citations:")
        for c in citations:
            path = c.get("path", "")
            start = c.get("start_line")
            end = c.get("end_line")
            repo = c.get("repo", "")
            line_str = (
                f":{start}-{end}"
                if start and end and start != end
                else (f":{start}" if start else "")
            )
            repo_str = f" ({repo})" if repo else ""
            print(f"  - `{path}{line_str}`{repo_str}")
        print("-" * 80)

    return 0


def handle_index(args: argparse.Namespace) -> int:
    """Index a GitHub repository into the vector store."""
    try:
        owner, repo = parse_repo_arg(args.repo)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            res = client.index_repo(
                owner=owner,
                repo=repo,
                branch=args.branch,
                token=args.token,
                force=args.force,
                sync=True,
            )
        except ApiClientError as exc:
            print(f"Error indexing repository via API: {exc}", file=sys.stderr)
            return 1
    else:
        from src.storage.indexer import index_repository
        from src.storage.chroma_store import get_vector_store
        from src.registry.registry import get_repo_registry

        storage = get_vector_store()
        registry = get_repo_registry()
        try:
            res = index_repository(
                owner=owner,
                repo=repo,
                branch=args.branch,
                token=args.token,
                collection_name=args.collection,
                incremental=not args.no_incremental,
                force=args.force,
                vector_store=storage,
                registry=registry,
            )
        except Exception as exc:
            print(f"Error indexing repository '{owner}/{repo}': {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"Repository '{owner}/{repo}' indexing complete:")
    print("=" * 60)
    print(f"Branch:          {res.get('branch', args.branch)}")
    if res.get("commit_sha"):
        print(f"Commit SHA:      {res.get('commit_sha')}")
    print(f"Files Processed: {res.get('files_processed', res.get('filtered_files', 0))}")
    print(f"Chunks Stored:   {res.get('chunks_stored', 0)}")
    print(f"Collection:      {res.get('collection', '')}")
    if res.get("unchanged"):
        print("Notice:          Repository was unchanged since last indexing.")
    return 0


def handle_changes(args: argparse.Namespace) -> int:
    """Preview pending Git changes from remote repository without reindexing."""
    try:
        owner, repo = parse_repo_arg(args.repo)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            res = client.get_repo_changes(owner, repo, branch=args.branch, token=args.token)
        except ApiClientError as exc:
            print(f"Error fetching changes from API: {exc}", file=sys.stderr)
            return 1
    else:
        from src.ingestion.github_client import get_repo_tree, get_latest_commit_sha
        from src.ingestion.filters import filter_files
        from src.registry.incremental import detect_changes
        from src.registry.registry import get_repo_registry

        registry = get_repo_registry()
        meta = registry.get(owner, repo)
        target_branch = args.branch or (meta.branch if meta else "main")

        try:
            latest_sha = get_latest_commit_sha(owner, repo, args.token, branch=target_branch)
        except Exception as e:
            latest_sha = None

        try:
            raw_tree = get_repo_tree(owner, repo, args.token, branch=target_branch)
            filtered_tree = filter_files(raw_tree)
        except Exception as e:
            print(f"Error fetching tree from GitHub for '{owner}/{repo}': {e}", file=sys.stderr)
            return 1

        prev_shas = meta.file_shas if meta else {}
        changeset = detect_changes(filtered_tree, prev_shas)
        res = {
            "owner": owner,
            "repo": repo,
            "branch": target_branch,
            "current_commit_sha": meta.commit_sha if meta else None,
            "latest_commit_sha": latest_sha,
            "added_files": changeset.added_files,
            "modified_files": changeset.modified_files,
            "deleted_files": changeset.deleted_files,
            "unchanged_files": changeset.unchanged_files,
            "has_changes": changeset.has_changes,
            "total_files": changeset.total_files,
        }

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"Pending Git Changes for '{owner}/{repo}' (Branch: {res.get('branch', 'main')})")
    print("=" * 65)
    print(f"Indexed Commit:   {res.get('current_commit_sha') or 'None (Not Indexed)'}")
    print(f"Remote Commit:    {res.get('latest_commit_sha') or 'N/A'}")
    print(f"Has Changes:      {res.get('has_changes', False)}")
    print(f"Added Files:      {len(res.get('added_files', []))}")
    for f in res.get("added_files", [])[:5]:
        print(f"  + {f}")
    if len(res.get("added_files", [])) > 5:
        print(f"  ... and {len(res.get('added_files', [])) - 5} more")

    print(f"Modified Files:   {len(res.get('modified_files', []))}")
    for f in res.get("modified_files", [])[:5]:
        print(f"  ~ {f}")
    if len(res.get("modified_files", [])) > 5:
        print(f"  ... and {len(res.get('modified_files', [])) - 5} more")

    print(f"Deleted Files:    {len(res.get('deleted_files', []))}")
    for f in res.get("deleted_files", [])[:5]:
        print(f"  - {f}")
    if len(res.get("deleted_files", [])) > 5:
        print(f"  ... and {len(res.get('deleted_files', [])) - 5} more")

    print(f"Unchanged Files:  {len(res.get('unchanged_files', []))}")
    return 0


def handle_delete(args: argparse.Namespace) -> int:
    """Delete a repository from registry and purge its vector collection."""
    try:
        owner, repo = parse_repo_arg(args.repo)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.yes:
        try:
            confirm = input(
                f"Are you sure you want to delete '{owner}/{repo}' and purge its vector collection? [y/N]: "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nDeletion cancelled.")
            return 0

        if confirm.strip().lower() not in ("y", "yes"):
            print("Deletion cancelled.")
            return 0

    if args.api_url:
        from src.ui.client import ApiClient, ApiClientError

        client = ApiClient(base_url=args.api_url)
        try:
            res = client.delete_repo(owner, repo)
        except ApiClientError as exc:
            print(f"Error deleting repository via API: {exc}", file=sys.stderr)
            return 1
    else:
        from src.storage.chroma_store import get_vector_store
        from src.registry.registry import get_repo_registry

        registry = get_repo_registry()
        storage = get_vector_store()
        meta = registry.get(owner, repo)
        if meta is None:
            print(f"Error: Repository '{owner}/{repo}' not found in registry.", file=sys.stderr)
            return 1

        if meta.collection_name:
            try:
                storage.clear(meta.collection_name)
            except Exception as e:
                logger.warning(
                    f"Error clearing vector collection {meta.collection_name}: {e}"
                )

        deleted = registry.delete(owner, repo)
        if not deleted:
            print(
                f"Error: Failed to delete repository '{owner}/{repo}' from registry.",
                file=sys.stderr,
            )
            return 1
        res = {
            "success": True,
            "message": f"Repository '{owner}/{repo}' and collection '{meta.collection_name}' deleted.",
            "target": f"{owner}/{repo}",
        }

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"Successfully deleted repository '{owner}/{repo}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="AI Codebase Intelligence Platform - Developer CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Optional API base URL (e.g. http://localhost:8000) to execute commands against a running server",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="Commands",
        description="Available CLI operations",
    )

    # status
    p_status = subparsers.add_parser(
        "status",
        help="Show platform status, storage stats, and active LLM configuration",
    )
    p_status.add_argument("--json", action="store_true", help="Output status summary as JSON")

    # repos / list
    p_repos = subparsers.add_parser(
        "repos",
        aliases=["list"],
        help="List all registered and indexed codebases",
    )
    p_repos.add_argument("--json", action="store_true", help="Output repository list as JSON")

    # query
    p_query = subparsers.add_parser(
        "query",
        help="Query indexed codebases with grounded citations",
    )
    p_query.add_argument("question", help="Natural language question about the codebase")
    p_query.add_argument("--repo", default=None, help="Target repository (owner/repo)")
    p_query.add_argument(
        "--repos",
        default=None,
        help="Comma-separated target repositories for cross-repo search",
    )
    p_query.add_argument("--collection", default=None, help="Vector store collection name override")
    p_query.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    p_query.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum prompt token budget for code context",
    )
    p_query.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable lexical & definition re-ranking",
    )
    p_query.add_argument(
        "--llm",
        default=None,
        choices=["fake", "gemini"],
        help="Override LLM provider",
    )
    p_query.add_argument("--json", action="store_true", help="Output complete response as JSON")

    # index
    p_index = subparsers.add_parser(
        "index",
        help="Index a GitHub repository into the vector database",
    )
    p_index.add_argument(
        "repo",
        help="Target GitHub repository (owner/repo, e.g. miguelgrinberg/microblog)",
    )
    p_index.add_argument("--branch", default="main", help="Git branch to index")
    p_index.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if commit SHA is unchanged",
    )
    p_index.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental indexing and index from scratch",
    )
    p_index.add_argument("--token", default=None, help="GitHub personal access token override")
    p_index.add_argument("--collection", default=None, help="Custom collection name")
    p_index.add_argument("--json", action="store_true", help="Output indexing result as JSON")

    # changes
    p_changes = subparsers.add_parser(
        "changes",
        help="Preview pending remote Git changes against local index",
    )
    p_changes.add_argument("repo", help="Target GitHub repository (owner/repo)")
    p_changes.add_argument(
        "--branch",
        default=None,
        help="Git branch to check (defaults to indexed branch)",
    )
    p_changes.add_argument("--token", default=None, help="GitHub personal access token override")
    p_changes.add_argument("--json", action="store_true", help="Output changeset as JSON")

    # delete
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete a repository and purge its vector collection",
    )
    p_delete.add_argument("repo", help="Target repository (owner/repo)")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_delete.add_argument("--json", action="store_true", help="Output deletion result as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if not args.subcommand:
        parser.print_help()
        return 0

    handlers = {
        "status": handle_status,
        "repos": handle_repos,
        "list": handle_repos,
        "query": handle_query,
        "index": handle_index,
        "changes": handle_changes,
        "delete": handle_delete,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
