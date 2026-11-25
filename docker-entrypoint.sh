#!/usr/bin/env bash
set -e

# Default port and logging configuration
API_PORT="${PORT:-8000}"
UI_PORT="${PORT:-8501}"
LOG_LEVEL_LOWER="$(printf '%s' "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"

# Handle commands
case "$1" in
    api)
        echo "Starting AI Codebase Intelligence FastAPI backend on port ${API_PORT}..."
        exec uvicorn src.api.app:app --host 0.0.0.0 --port "${API_PORT}" --no-access-log --log-level "${LOG_LEVEL_LOWER}"
        ;;
    ui)
        echo "Starting AI Codebase Intelligence Streamlit UI on port ${UI_PORT}..."
        exec streamlit run src/ui/app.py --server.port "${UI_PORT}" --server.address 0.0.0.0 --server.headless true
        ;;
    eval)
        shift
        echo "Running evaluation suite..."
        exec python -m src.evaluation.main "$@"
        ;;
    cli)
        shift
        echo "Running developer CLI tool..."
        exec python -m src.cli.main "$@"
        ;;
    audit)
        shift
        echo "Running ingestion audit check..."
        exec python -m src.cli.main audit "$@"
        ;;
    doctor)
        shift
        echo "Running platform diagnostic doctor check..."
        exec python -m src.cli.main doctor "$@"
        ;;
    test)
        shift
        echo "Running test suite..."
        exec pytest "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
