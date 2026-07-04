import re
from parsers.base import BaseParser

try:
    import javalang
    _HAS_JAVALANG = True
except ImportError:
    _HAS_JAVALANG = False


class JavaParser(BaseParser):
    """
    Parses Java source.

    Primary path uses `javalang` (pip install javalang) for accurate AST
    extraction. If javalang isn't available, falls back to a lightweight
    regex parser that still captures classes, methods, and imports —
    enough for a working demo, though less precise on call resolution.
    """

    def extract_entities(self, source: str, file_path: str) -> list[dict]:
        if _HAS_JAVALANG:
            return self._parse_with_javalang(source, file_path)
        return self._parse_with_regex(source, file_path)

    # ==================================================================
    # Primary path — javalang AST
    # ==================================================================

    def _parse_with_javalang(self, source: str, file_path: str) -> list[dict]:
        tree = javalang.parse.parse(source)

        imports  = [imp.path for imp in tree.imports]
        is_test  = self.is_test_file(file_path)
        lines    = source.splitlines()
        entities = []

        for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = class_node.name
            class_start = class_node.position.line if class_node.position else 0

            # Class-level entity
            entities.append({
                "entity_id":  self.make_entity_id(file_path, class_name, class_start),
                "type":       "class",
                "file_path":  file_path,
                "line_start": class_start,
                "line_end":   self._estimate_class_end(class_node, lines, class_start),
                "calls":      self._collect_class_calls(class_node),
                "imports":    imports,
                "has_tests":  False if is_test else False,
                "raw_source": self._slice_source(lines, class_start,
                                                  self._estimate_class_end(class_node, lines, class_start)),
            })

            # Methods within the class
            for method in class_node.methods:
                m_start = method.position.line if method.position else class_start
                m_name  = f"{class_name}.{method.name}"
                m_end   = self._estimate_method_end(method, lines, m_start)

                entities.append({
                    "entity_id":  self.make_entity_id(file_path, m_name, m_start),
                    "type":       "method",
                    "file_path":  file_path,
                    "line_start": m_start,
                    "line_end":   m_end,
                    "calls":      self._collect_method_calls(method),
                    "imports":    imports,
                    "has_tests":  False if is_test else False,
                    "raw_source": self._slice_source(lines, m_start, m_end),
                })

        # Interfaces
        for _, iface in tree.filter(javalang.tree.InterfaceDeclaration):
            i_start = iface.position.line if iface.position else 0
            entities.append({
                "entity_id":  self.make_entity_id(file_path, iface.name, i_start),
                "type":       "interface",
                "file_path":  file_path,
                "line_start": i_start,
                "line_end":   i_start,
                "calls":      [],
                "imports":    imports,
                "has_tests":  False,
                "raw_source": self._slice_source(lines, i_start, i_start + 5),
            })

        return entities

    def _collect_method_calls(self, method) -> list[str]:
        """Walk a method node and collect invoked method names."""
        calls = []
        for _, node in method.filter(javalang.tree.MethodInvocation):
            calls.append(node.member)
            if node.qualifier:
                calls.append(f"{node.qualifier}.{node.member}")
        return self._dedupe(calls)

    def _collect_class_calls(self, class_node) -> list[str]:
        """Collect all method invocations anywhere in the class."""
        calls = []
        for _, node in class_node.filter(javalang.tree.MethodInvocation):
            calls.append(node.member)
            if node.qualifier:
                calls.append(f"{node.qualifier}.{node.member}")
        return self._dedupe(calls)

    def _estimate_method_end(self, method, lines, start) -> int:
        """
        javalang doesn't give end positions, so estimate by brace matching
        from the method's start line.
        """
        return self._match_braces(lines, start)

    def _estimate_class_end(self, class_node, lines, start) -> int:
        return self._match_braces(lines, start)

    # ==================================================================
    # Fallback path — regex (when javalang unavailable)
    # ==================================================================

    def _parse_with_regex(self, source: str, file_path: str) -> list[dict]:
        lines    = source.splitlines()
        imports  = re.findall(r"^\s*import\s+([\w.]+);", source, re.MULTILINE)
        entities = []

        # Classes
        class_pattern = re.compile(
            r"(?:public|private|protected)?\s*(?:abstract\s+|final\s+)?class\s+(\w+)"
        )
        # Methods (rough): modifier returntype name(...)
        method_pattern = re.compile(
            r"(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{"
        )

        for i, line in enumerate(lines, start=1):
            cm = class_pattern.search(line)
            if cm:
                name = cm.group(1)
                end  = self._match_braces(lines, i)
                entities.append({
                    "entity_id":  self.make_entity_id(file_path, name, i),
                    "type":       "class",
                    "file_path":  file_path,
                    "line_start": i,
                    "line_end":   end,
                    "calls":      self._regex_calls(self._slice_source(lines, i, end)),
                    "imports":    imports,
                    "has_tests":  False,
                    "raw_source": self._slice_source(lines, i, end),
                })

            mm = method_pattern.search(line)
            if mm:
                name = mm.group(1)
                if name in ("if", "for", "while", "switch", "catch"):
                    continue  # control keywords match the pattern — skip
                end = self._match_braces(lines, i)
                entities.append({
                    "entity_id":  self.make_entity_id(file_path, name, i),
                    "type":       "method",
                    "file_path":  file_path,
                    "line_start": i,
                    "line_end":   end,
                    "calls":      self._regex_calls(self._slice_source(lines, i, end)),
                    "imports":    imports,
                    "has_tests":  False,
                    "raw_source": self._slice_source(lines, i, end),
                })

        return entities

    def _regex_calls(self, text: str) -> list[str]:
        """Crude call detection: word followed by an opening paren."""
        raw = re.findall(r"(\w+)\s*\(", text)
        keywords = {"if", "for", "while", "switch", "catch", "return", "new"}
        return self._dedupe([c for c in raw if c not in keywords])

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _match_braces(self, lines, start_line) -> int:
        """
        Find the line where the block opened at start_line closes,
        by counting { and } from that line onward.
        Returns the closing line number (1-indexed).
        """
        depth = 0
        started = False
        for i in range(start_line - 1, len(lines)):
            depth += lines[i].count("{")
            depth -= lines[i].count("}")
            if "{" in lines[i]:
                started = True
            if started and depth <= 0:
                return i + 1
        return len(lines)

    def _slice_source(self, lines, start, end) -> str:
        """Return source text between two 1-indexed line numbers, inclusive."""
        start = max(1, start)
        end   = min(len(lines), end)
        return "\n".join(lines[start - 1:end])

    @staticmethod
    def _dedupe(items) -> list[str]:
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out