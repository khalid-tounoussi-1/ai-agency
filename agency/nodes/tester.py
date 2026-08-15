"""Tester -- the one node with no model in it.

It compiles every source file, then runs pytest as a subprocess and reports
what actually happened. Nothing here asks an LLM whether the code looks right.
A run's success is decided by executed assertions, and only by those.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import config
from ..state import AgencyState

MAX_OUTPUT_CHARS = 6000

_PASSED = re.compile(r"(\d+) passed")
_FAILED = re.compile(r"(\d+) failed")
_ERRORS = re.compile(r"(\d+) errors?")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2 :]
    return f"{head}\n...[{len(text) - MAX_OUTPUT_CHARS} chars elided]...\n{tail}"


def _syntax_check(root: Path, files: dict[str, str]) -> str | None:
    """Cheap precise failure before paying for a pytest process."""
    for rel, content in files.items():
        if not rel.endswith(".py"):
            continue
        try:
            compile(content, rel, "exec")
        except SyntaxError as exc:
            return f"{rel}:{exc.lineno}: {exc.__class__.__name__}: {exc.msg}\n  {(exc.text or '').rstrip()}"
    return None


def tester_node(state: AgencyState) -> dict[str, Any]:
    root = Path(state["workspace"])
    files = state.get("files") or {}
    test_path = (state.get("plan") or {}).get("test_file", "the test file")

    syntax_error = _syntax_check(root, files)
    if syntax_error:
        report = {
            "ok": False,
            "stage": "syntax",
            "passed": 0,
            "failed": 0,
            "output": syntax_error,
        }
        return {
            "test_report": report,
            "events": [{"node": "tester", "stage": "syntax", "ok": False}],
        }

    # Anchors pytest's rootdir here so the workspace lands on sys.path and
    # `import mypkg` resolves to the generated code.
    conftest = root / "conftest.py"
    if not conftest.exists():
        conftest.write_text("")

    # Without an explicit config, pytest searches upward from the workspace and
    # can adopt the config of whatever repo the agency happens to be running
    # inside -- whose testpaths and norecursedirs have nothing to do with the
    # generated project, and silently collected zero tests here. `-c` pins both
    # the config and the rootdir to this workspace.
    ini = root / "pytest.ini"
    ini.write_text("[pytest]\ntestpaths = .\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH", "")]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "-c",
                str(ini),
                "--rootdir",
                str(root),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=config.TEST_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        report = {
            "ok": False,
            "stage": "timeout",
            "passed": 0,
            "failed": 0,
            "output": f"pytest exceeded {config.TEST_TIMEOUT}s and was killed. "
            "Look for an infinite loop or unbounded recursion.",
        }
        return {
            "test_report": report,
            "events": [{"node": "tester", "stage": "timeout", "ok": False}],
        }

    output = _truncate(f"{proc.stdout}\n{proc.stderr}".strip())
    passed = int(m.group(1)) if (m := _PASSED.search(output)) else 0
    failed = int(m.group(1)) if (m := _FAILED.search(output)) else 0
    errors = int(m.group(1)) if (m := _ERRORS.search(output)) else 0

    if proc.returncode == 5:  # pytest's "no tests collected"
        # Distinct from a failing suite: no amount of rewriting the source will
        # help, so it is reported as its own stage and routed differently.
        ok, stage = False, "collection"
        output = f"pytest collected no tests from {test_path}.\n{output}"
    else:
        ok, stage = (proc.returncode == 0 and passed > 0), "pytest"

    report = {
        "ok": ok,
        "stage": stage,
        "passed": passed,
        "failed": failed + errors,
        "output": output,
    }
    return {
        "test_report": report,
        "events": [
            {
                "node": "tester",
                "stage": stage,
                "ok": ok,
                "passed": passed,
                "failed": failed + errors,
            }
        ],
    }
