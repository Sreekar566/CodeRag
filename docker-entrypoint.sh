#!/usr/bin/env bash
set -e

# Default port configuration
API_PORT="${PORT:-8000}"
UI_PORT="${PORT:-8501}"

# Handle commands
case "$1" in
    api)
        echo "Starting AI Codebase Intelligence FastAPI backend on port ${API_PORT}..."
        exec uvicorn src.api.app:app --host 0.0.0.0 --port "${API_PORT}"
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
    test)
        shift
        echo "Running test suite..."
        exec pytest "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
