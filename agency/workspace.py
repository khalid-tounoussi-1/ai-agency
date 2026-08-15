"""A run's writable directory, and the only path validation in the system.

Nodes never touch the filesystem directly -- they hand a relative path and
content to a Workspace, which refuses anything that would escape the run root.
"""
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
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
