"""A run's writable directory, and the only path validation in the system.

Nodes never touch the filesystem directly -- they hand a relative path and
content to a Workspace, which refuses anything that would escape the run root.
"""
import ast
from pathlib import Path


class PathRejected(ValueError):
    pass


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        candidate = Path(rel)
        if candidate.is_absolute():
            raise PathRejected(f"absolute paths are not allowed: {rel!r}")
        if ".." in candidate.parts:
            raise PathRejected(f"parent traversal is not allowed: {rel!r}")
        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise PathRejected(f"path escapes the workspace: {rel!r}")
        return target

    def write(self, rel: str, content: str) -> Path:
        target = self.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text()

    def exists(self, rel: str) -> bool:
        return self.resolve(rel).exists()

    def list_files(self) -> list[str]:
        skip = {"__pycache__", ".git", ".venv", ".agency"}
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and not skip & set(p.parts)
        )

    def manifest(self) -> dict[str, list[str]]:
        """Every Python file already here, with its top-level names. Compact
        enough to put in a prompt, which a full listing of contents is not."""
        found: dict[str, list[str]] = {}
        for rel in self.list_files():
            if not rel.endswith(".py"):
                continue
            try:
                tree = ast.parse(self.read(rel))
            except (SyntaxError, UnicodeDecodeError):
                found[rel] = []
                continue
            found[rel] = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
        return found
