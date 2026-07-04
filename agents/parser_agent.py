import os
import hashlib
from datetime import datetime, timezone
from typing import Any

from parsers.python_parser import PythonParser
from parsers.java_parser import JavaParser
from db.schema import ENTITIES, METADATA

# ---------------------------------------------------------------------------
# Language router
# ---------------------------------------------------------------------------

PARSER_MAP = {
    "python": (PythonParser, ".py"),
    "java":   (JavaParser,   ".java"),
}


def _get_parser(language: str):
    if language not in PARSER_MAP:
        raise ValueError(f"Unsupported language: {language}. Choose from {list(PARSER_MAP)}")
    cls, ext = PARSER_MAP[language]
    return cls(), ext


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".git", ".github", "__pycache__", "node_modules",
    ".venv", "venv", "env", "build", "dist", "target",
    ".idea", ".vscode",
}

def _discover_files(repo_path: str, ext: str) -> list[str]:
    """
    Walk the repo and return all file paths matching the given extension.
    Skips common non-source directories.
    """
    matched = []
    for root, dirs, files in os.walk(repo_path):
        # Prune directories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.endswith(ext):
                matched.append(os.path.join(root, fname))
    return matched


# ---------------------------------------------------------------------------
# Repo identity
# ---------------------------------------------------------------------------

def _repo_id(repo_path: str) -> str:
    """
    Stable identifier for a repo — hash of its absolute path.
    Used as the key in the metadata collection.
    """
    return hashlib.md5(os.path.abspath(repo_path).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Parser agent node
# ---------------------------------------------------------------------------

def parser_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — P1 ingestion pipeline, step 1.

    Reads:
        state["repo_path"]  — absolute or relative path to the repo root
        state["language"]   — "python" or "java"
        state["db"]         — live MongoDB database handle (injected by pipeline)

    Writes back to state:
        state["entities"]      — list of entity dicts extracted from source
        state["parse_errors"]  — list of files that failed to parse
        state["repo_id"]       — stable repo identifier
        state["repo_path"]     — unchanged, passed through
        state["language"]      — unchanged, passed through
        state["db"]            — unchanged, passed through
    """

    repo_path = state["repo_path"]
    language  = state["language"]
    db        = state["db"]

    parser, ext = _get_parser(language)
    repo_id     = _repo_id(repo_path)

    print(f"[parser_agent] scanning {repo_path!r} for {ext} files...")

    files = _discover_files(repo_path, ext)
    print(f"[parser_agent] {len(files)} files found")

    entities: list[dict] = []
    parse_errors: list[str] = []

    for fpath in files:
        rel_path = os.path.relpath(fpath, repo_path)
        try:
            source = open(fpath, encoding="utf-8", errors="ignore").read()
            extracted = parser.extract_entities(source, rel_path)
            entities.extend(extracted)
        except Exception as exc:
            parse_errors.append(f"{rel_path}: {exc}")
            print(f"[parser_agent] WARN — skipped {rel_path}: {exc}")

    print(f"[parser_agent] extracted {len(entities)} entities, {len(parse_errors)} errors")

    # Stamp repo_id onto every entity so downstream agents can filter by repo.
    # The parser itself doesn't know the repo_id, so we add it here.
    for e in entities:
        e["repo_id"] = repo_id

    # ------------------------------------------------------------------
    # Persist to MongoDB
    # ------------------------------------------------------------------
    if entities:
        col = db[ENTITIES]
        # Replace only THIS repo's entities, leaving other ingested repos
        # intact, so multiple codebases can coexist and be switched between.
        col.delete_many({"repo_id": repo_id})
        col.insert_many(entities)

        # Uniqueness is per-repo (repo_id + entity_id), not global — two
        # different repos may legitimately share an entity_id like
        # "main.py::main::1". A global unique index would wrongly reject that.
        col.create_index([("repo_id", 1), ("entity_id", 1)], unique=True)
        col.create_index("file_path")
        col.create_index("calls")
        col.create_index("repo_id")

        print(f"[parser_agent] wrote {len(entities)} documents for repo {repo_id}")

    # ------------------------------------------------------------------
    # Update metadata record
    # ------------------------------------------------------------------
    db[METADATA].update_one(
        {"repo_id": repo_id},
        {"$set": {
            "repo_id":        repo_id,
            "repo_path":      os.path.abspath(repo_path),
            "language":       language,
            "entity_count":   len(entities),
            "parse_errors":   parse_errors,
            "parser_done":    True,
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }},
        upsert=True,
    )

    return {
        **state,
        "repo_id":      repo_id,
        "entities":     entities,
        "parse_errors": parse_errors,
    }