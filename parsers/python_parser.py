import ast
from parsers.base import BaseParser


class PythonParser(BaseParser):
    """
    Parses Python source using the standard library `ast` module.
    Fully deterministic — no LLM, no external dependencies.
    """

    def extract_entities(self, source: str, file_path: str) -> list[dict]:
        # ast.parse turns source text into an Abstract Syntax Tree.
        # A SyntaxError here means the file isn't valid Python —
        # the caller (parser_agent) catches it and logs a parse error.
        tree = ast.parse(source)

        imports   = self._extract_imports(tree)
        is_test   = self.is_test_file(file_path)
        entities  = []

        # Walk only the top-level + nested defs we care about.
        # We handle classes specially so their methods get qualified names.
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entities.append(
                    self._build_class_entity(node, source, file_path, imports, is_test)
                )
                # Methods inside the class
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        entities.append(
                            self._build_function_entity(
                                child, source, file_path, imports, is_test,
                                qualifier=node.name,
                            )
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods — already handled inside their class above.
                if self._is_method(node, tree):
                    continue
                entities.append(
                    self._build_function_entity(
                        node, source, file_path, imports, is_test
                    )
                )

        return entities

    # ------------------------------------------------------------------
    # Entity builders
    # ------------------------------------------------------------------

    def _build_function_entity(self, node, source, file_path, imports, is_test, qualifier=None):
        name = node.name if qualifier is None else f"{qualifier}.{node.name}"
        calls = self._extract_calls(node)
        raw   = ast.get_source_segment(source, node) or ""

        return {
            "entity_id":  self.make_entity_id(file_path, name, node.lineno),
            "type":       "method" if qualifier else "function",
            "file_path":  file_path,
            "line_start": node.lineno,
            "line_end":   getattr(node, "end_lineno", node.lineno),
            "calls":      calls,
            "imports":    imports,
            "has_tests":  False if is_test else self._guess_has_tests(name, file_path),
            "raw_source": raw,
        }

    def _build_class_entity(self, node, source, file_path, imports, is_test):
        name  = node.name
        calls = self._extract_calls(node)   # calls anywhere in the class body
        raw   = ast.get_source_segment(source, node) or ""

        return {
            "entity_id":  self.make_entity_id(file_path, name, node.lineno),
            "type":       "class",
            "file_path":  file_path,
            "line_start": node.lineno,
            "line_end":   getattr(node, "end_lineno", node.lineno),
            "calls":      calls,
            "imports":    imports,
            "has_tests":  False if is_test else self._guess_has_tests(name, file_path),
            "raw_source": raw,
        }

    # ------------------------------------------------------------------
    # AST extraction helpers
    # ------------------------------------------------------------------

    def _extract_calls(self, node) -> list[str]:
        """
        Walk a node's subtree and collect the names of everything it calls.
        Handles two forms:
            foo()             -> "foo"        (ast.Name)
            obj.method()      -> "method"     (ast.Attribute)
            module.func()     -> "func"
        Deduplicated, order-stable.
        """
        calls = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    calls.append(fn.id)
                elif isinstance(fn, ast.Attribute):
                    # For obj.method(), capture both "method" and "obj.method"
                    calls.append(fn.attr)
                    full = self._attribute_to_str(fn)
                    if full and full != fn.attr:
                        calls.append(full)
        # Deduplicate while preserving order
        seen, out = set(), []
        for c in calls:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _attribute_to_str(self, node) -> str:
        """
        Convert an ast.Attribute chain back to dotted string.
        e.g. self.token_service.tokenize -> "self.token_service.tokenize"
        """
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    def _extract_imports(self, tree) -> list[str]:
        """Collect all import statements as flat strings."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports

    def _is_method(self, func_node, tree) -> bool:
        """
        Determine whether a FunctionDef is a method (lives inside a class).
        We check by walking classes and seeing if this node is in their body.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False

    def _guess_has_tests(self, name, file_path) -> bool:
        """
        Placeholder heuristic. In the real pipeline, parser_agent does a
        second pass scanning test files for references to this name.
        Here we conservatively return False; the agent fills this in.
        """
        return False