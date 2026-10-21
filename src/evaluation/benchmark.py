"""
Curated benchmark datasets and dataset loaders for Codebase Intelligence evaluation.
Includes standard benchmark test cases for miguelgrinberg/microblog covering authentication,
database schemas, application factories, error handlers, and adversarial negative probes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.models import BenchmarkItem

DEFAULT_MICROBLOG_BENCHMARK: list[dict[str, Any]] = [
    {
        "id": "microblog-auth-login",
        "question": "How is user authentication and the login route implemented?",
        "expected_files": ["app/auth/routes.py", "app/auth/forms.py"],
        "expected_symbols": ["login", "LoginForm", "current_user", "redirect"],
        "ground_truth": "User authentication is implemented in app/auth/routes.py via the login view. It validates LoginForm, verifies password via check_password, and logs in using Flask-Login.",
        "category": "authentication",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-auth-register",
        "question": "Where is user registration handled and how are new users created?",
        "expected_files": ["app/auth/routes.py", "app/auth/forms.py"],
        "expected_symbols": ["register", "RegistrationForm", "set_password", "db.session"],
        "ground_truth": "Registration is handled in app/auth/routes.py under register route using RegistrationForm, hashing passwords via set_password, and committing to db.session.",
        "category": "authentication",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-auth-reset-email",
        "question": "How does password reset email generation and token verification work?",
        "expected_files": ["app/auth/email.py", "app/auth/routes.py"],
        "expected_symbols": ["send_password_reset_email", "reset_password_request", "get_reset_password_token"],
        "ground_truth": "Password reset is requested in app/auth/routes.py, which generates tokens and sends emails via send_password_reset_email in app/auth/email.py.",
        "category": "authentication",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-models-user",
        "question": "What fields and methods are defined on the User model?",
        "expected_files": ["app/models.py"],
        "expected_symbols": ["User", "username", "email", "password_hash", "set_password", "check_password"],
        "ground_truth": "The User model in app/models.py inherits from db.Model and UserMixin, defining username, email, password_hash, and password methods.",
        "category": "models",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-models-post",
        "question": "How is the Post model structured and how does it relate to User?",
        "expected_files": ["app/models.py"],
        "expected_symbols": ["Post", "body", "timestamp", "user_id", "author"],
        "ground_truth": "Post model in app/models.py stores posts with body, timestamp, and user_id foreign key, linked back to User via relationship.",
        "category": "models",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-models-followers",
        "question": "How does the following and followers social relationship work?",
        "expected_files": ["app/models.py"],
        "expected_symbols": ["followers", "followed", "follow", "unfollow", "is_following"],
        "ground_truth": "Followers relationship in app/models.py is an association table followers with follow, unfollow, and is_following helper methods.",
        "category": "models",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-app-factory",
        "question": "How is the Flask application initialized using the application factory pattern?",
        "expected_files": ["app/__init__.py", "microblog.py"],
        "expected_symbols": ["create_app", "Flask", "register_blueprint"],
        "ground_truth": "The create_app function in app/__init__.py initializes Flask extensions (db, migrate, login, mail) and registers blueprints, invoked in microblog.py.",
        "category": "factory",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-config",
        "question": "Where are configuration settings like SECRET_KEY and database URLs configured?",
        "expected_files": ["config.py"],
        "expected_symbols": ["Config", "SECRET_KEY", "SQLALCHEMY_DATABASE_URI"],
        "ground_truth": "Config class in config.py loads SECRET_KEY and SQLALCHEMY_DATABASE_URI from environment variables with fallbacks.",
        "category": "configuration",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-errors",
        "question": "How are HTTP 404 and 500 errors handled in the application?",
        "expected_files": ["app/errors/handlers.py"],
        "expected_symbols": ["not_found_error", "internal_error", "404", "500"],
        "ground_truth": "Error handlers are defined in app/errors/handlers.py decorating 404 and 500 error pages and rolling back db session on 500.",
        "category": "error_handling",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-main-explore",
        "question": "Where is the explore feed route and how does it paginate posts?",
        "expected_files": ["app/main/routes.py"],
        "expected_symbols": ["explore", "paginate", "POSTS_PER_PAGE"],
        "ground_truth": "The explore route is in app/main/routes.py, querying posts ordered by timestamp descending and paginating with paginate.",
        "category": "routes",
        "is_negative": False,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-neg-stripe",
        "question": "Where is the Stripe payment processing and checkout webhook implemented?",
        "expected_files": [],
        "expected_symbols": [],
        "ground_truth": "Stripe payment processing is not implemented in microblog; this negative probe tests hallucination abstention.",
        "category": "negative_probe",
        "is_negative": True,
        "target_repo": "miguelgrinberg/microblog",
    },
    {
        "id": "microblog-neg-k8s",
        "question": "Where are the Kubernetes deployment manifests and Helm charts configured?",
        "expected_files": [],
        "expected_symbols": [],
        "ground_truth": "Kubernetes and Helm are not configured in microblog; this negative probe tests hallucination abstention.",
        "category": "negative_probe",
        "is_negative": True,
        "target_repo": "miguelgrinberg/microblog",
    },
]


def get_default_benchmark() -> list[BenchmarkItem]:
    """Retrieve the standard benchmark item suite as BenchmarkItem dataclasses."""
    return [BenchmarkItem.from_dict(item) for item in DEFAULT_MICROBLOG_BENCHMARK]


def save_benchmark_to_json(filepath: str | Path, items: list[BenchmarkItem] | None = None) -> None:
    """Save benchmark items to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_list = items if items is not None else get_default_benchmark()
    raw = [item.to_dict() for item in benchmark_list]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def load_benchmark_from_json(filepath: str | Path) -> list[BenchmarkItem]:
    """Load benchmark items from a JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of benchmark items in {filepath}, got {type(data).__name__}")
    return [BenchmarkItem.from_dict(item) for item in data]
