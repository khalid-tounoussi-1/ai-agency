"""Planner -- turns a task into a build plan and a set of acceptance tests.

The acceptance tests are the important half. They are written here, from the
task, before any implementation exists, so the contract the Coder must satisfy
is derived from the requirement rather than from the code that happens to get
written. That ordering is what stops the agency from grading its own homework.
"""
from typing import Any

from .. import llm
from ..state import AgencyState

SYSTEM = (
    "You are the Planner of a small software agency. You do not write "
    "implementation code. You produce a build plan as a single JSON object."
)

SCHEMA = """Return a single JSON object with exactly these keys:

  "summary"           one sentence describing what will be built.
  "files"             array of {"path", "purpose"} for SOURCE files only.
                      Relative paths, no "..", Python only. Prefer 1-3 files.
                      Never put a test file in here.
  "test_file"         relative path for the pytest file, e.g. "tests/test_x.py".
  "acceptance_tests"  array of {"name", "asserts"}. One entry per behaviour the
                      TASK requires. "name" is a pytest function name starting
                      with "test_". "asserts" is a CONCRETE Python expression
                      that must evaluate true, e.g.
                      "flatten([1, [2, [3]]]) == [1, 2, 3]".
                      Cover the edge cases the task implies, not just the happy
                      path. Aim for 5-10 entries.
  "notes"             constraints the Coder must respect (standard library
                      only, no network, no file I/O unless the task asks).

Rules: source files import only the Python standard library. Every acceptance
test must be mechanically checkable -- no assertions about style or intent."""


def _validate(data: dict[str, Any]) -> None:
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError('"files" must be a non-empty array of {"path","purpose"}')
    for f in files:
        if not isinstance(f, dict) or "path" not in f:
            raise ValueError('every entry in "files" needs a "path"')
        path = str(f["path"])
        if path.startswith("/") or ".." in path:
            raise ValueError(f"file path must be relative and contain no '..': {path!r}")
        if not path.endswith(".py"):
            raise ValueError(f"source files must be Python (.py): {path!r}")
        if path.startswith("tests/") or path.split("/")[-1].startswith("test_"):
            raise ValueError(f'"files" is for source only; {path!r} looks like a test file')

    test_file = str(data.get("test_file", ""))
    if not test_file.endswith(".py") or test_file.startswith("/") or ".." in test_file:
        raise ValueError('"test_file" must be a relative .py path such as "tests/test_x.py"')
    if not (test_file.startswith("tests/") or test_file.split("/")[-1].startswith("test_")):
        raise ValueError('"test_file" must be named so pytest collects it (tests/test_*.py)')

    accepts = data.get("acceptance_tests")
    if not isinstance(accepts, list) or len(accepts) < 2:
        raise ValueError('"acceptance_tests" must contain at least 2 concrete cases')
    seen = set()
    for a in accepts:
        if not isinstance(a, dict) or not a.get("name") or not a.get("asserts"):
            raise ValueError('every acceptance test needs both "name" and "asserts"')
        name = str(a["name"])
        if not name.startswith("test_"):
            raise ValueError(f"acceptance test name must start with 'test_': {name!r}")
        if name in seen:
            raise ValueError(f"duplicate acceptance test name: {name!r}")
        seen.add(name)


def planner_node(state: AgencyState) -> dict[str, Any]:
    replans = state.get("replans", 0)
    previous = state.get("plan")
    review = state.get("review") or {}

    if previous and review.get("test_file_defective"):
        user = (
            f"TASK:\n{state['task']}\n\n"
            f"Your previous plan produced a test suite the Reviewer judged defective.\n"
            f"Reviewer's reason: {review.get('defect_reason') or review.get('diagnosis')}\n\n"
            f"Previous acceptance tests:\n"
            + "\n".join(f"  - {a['name']}: {a['asserts']}" for a in previous["acceptance_tests"])
            + "\n\nProduce a corrected plan. Fix the faulty acceptance tests; keep the "
            "ones that were sound.\n\n" + SCHEMA
        )
        replans += 1
    else:
        user = f"TASK:\n{state['task']}\n\n{SCHEMA}"

    plan = llm.ask_json("planner", SYSTEM, user, validate=_validate)
    plan.setdefault("summary", "")
    plan.setdefault("notes", "")

    return {
        "plan": plan,
        "replans": replans,
        # A new plan invalidates the previous verdict and repair list.
        "review": {},
        "events": [
            {
                "node": "planner",
                "replan": replans > state.get("replans", 0),
                "summary": plan["summary"],
                "files": [f["path"] for f in plan["files"]],
                "test_file": plan["test_file"],
                "acceptance_tests": len(plan["acceptance_tests"]),
            }
        ],
    }
