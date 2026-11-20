"""
Execution entrypoint for 'python -m src.cli'.
"""
import sys
from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
