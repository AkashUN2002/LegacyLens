from abc import ABC, abstractmethod
import os
import re


class BaseParser(ABC):
    """
    Abstract base class for all language parsers.

    A parser's single job: take the source text of one file and return
    a list of entity dicts. It does NOT touch MongoDB, LangGraph, or any LLM.

    Every entity dict must contain at least these keys:
        entity_id   : str   — stable unique identifier
        type        : str   — "function" | "class" | "method" | "interface"
        file_path   : str   — path relative to repo root
        line_start  : int
        line_end    : int
        calls       : list[str]  — names of things this entity calls
        imports     : list[str]  — import statements in the file
        has_tests   : bool       — heuristic: is this entity referenced by a test?
        raw_source  : str        — the entity's source text
    """

    # ------------------------------------------------------------------
    # The one method every subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_entities(self, source: str, file_path: str) -> list[dict]:
        """
        Parse one file's source text into a list of entity dicts.

        Args:
            source:    the full text content of the file
            file_path: path relative to the repo root (for entity_id + citations)

        Returns:
            list of entity dicts conforming to the schema above.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def make_entity_id(file_path: str, name: str, line: int | None = None) -> str:
        """
        Build a stable, unique entity_id from the file path and entity name.
        Example: "src/payments/Processor.java" + "processCard"
                 -> "src/payments/Processor.java::processCard"

        A `line` number is appended when provided so that two entities with
        the same name in the same file (e.g. an overload, a redefinition, or
        a nested function sharing a parent's name) get distinct ids and don't
        collide on the unique index.

        Backslashes are normalised to forward slashes so Windows and POSIX
        paths produce identical ids for the same file.
        """
        normalised = file_path.replace("\\", "/")
        base = f"{normalised}::{name}"
        return f"{base}::{line}" if line is not None else base

    @staticmethod
    def is_test_file(file_path: str) -> bool:
        """
        Heuristic: does this file look like a test file?
        Matches common conventions across Python, Java, and JS.
        """
        fname = os.path.basename(file_path).lower()
        patterns = [
            fname.startswith("test_"),
            fname.endswith("_test.py"),
            fname.endswith("test.java"),
            fname.endswith("tests.java"),
            fname.endswith(".spec.js"),
            fname.endswith(".test.js"),
            "/test/" in file_path.lower().replace("\\", "/"),
            "/tests/" in file_path.lower().replace("\\", "/"),
        ]
        return any(patterns)

    @staticmethod
    def name_referenced_in(name: str, text: str) -> bool:
        """
        Check whether `name` appears as a whole-word token in `text`.
        Used for the test-coverage heuristic — does any test file
        reference this entity by name?
        """
        return re.search(rf"\b{re.escape(name)}\b", text) is not None