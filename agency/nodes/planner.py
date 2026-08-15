"""Planner -- turns a task into a build plan and a set of acceptance tests.

The acceptance tests are the important half. They are written here, from the
task, before any implementation exists, so the contract the Coder must satisfy
is derived from the requirement rather than from the code that happens to get
written. That ordering is what stops the agency from grading its own homework.
"""
from pathlib import Path
from typing import Any

from .. import llm, testgen
from ..state import AgencyState
from ..workspace import Workspace

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
  "test_imports"      array of import lines the tests need, such as
                      ["from student_registry import StudentRegistry"].
                      pytest itself is imported for you.
  "acceptance_tests"  array of cases, one per behaviour the TASK requires.
                      Cover the edge cases it implies, not just the happy path.
                      Aim for 8-14 entries. Each case has:

                        "name"     pytest function name starting with "test_".
                        "setup"    optional array of Python STATEMENTS run
                                   first, in order.
                      and then EXACTLY ONE of:
                        "asserts"  a Python EXPRESSION that must be true.
                      or:
                        "raises"   an exception class name, and
                        "call"     the expression that must raise it.

  "notes"             constraints the Coder must respect (standard library
                      only, no network, no file I/O unless the task asks).

These are compiled as Python before anything is built, so write real Python:

  {"name": "test_counts_active_students",
   "setup": ["r = StudentRegistry()",
             "r.add_student('s1', 'Ada', 9, 'ada@school.org')"],
   "asserts": "r.count_active() == 1"}

  {"name": "test_rejects_blank_id",
   "setup": ["r = StudentRegistry()"],
   "raises": "ValueError",
   "call": "r.add_student('', 'Ada', 9, 'ada@school.org')"}

"r.add_student('') raises ValueError" is English, not an expression. It will be
rejected -- use the "raises"/"call" form for every error case.

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

    testgen.validate_imports(data.get("test_imports"))

    accepts = data.get("acceptance_tests")
    if not isinstance(accepts, list) or len(accepts) < 2:
        raise ValueError('"acceptance_tests" must contain at least 2 concrete cases')
    seen = set()
    for a in accepts:
        if not isinstance(a, dict):
            raise ValueError("every acceptance test must be a JSON object")
        # Compiles the case as Python. A test that will not parse is caught
        # here, where it costs one model call, rather than after the build.
        testgen.validate_case(a)
        name = str(a["name"])
        if name in seen:
            raise ValueError(f"duplicate acceptance test name: {name!r}")
        seen.add(name)


def _case_line(case: dict[str, Any]) -> str:
    return f"{case['name']}: {testgen.describe(case)}"


def _existing_project(state: AgencyState) -> str:
    """What is already in the target directory, so a second commission extends
    the project instead of colliding with it."""
    manifest = Workspace(Path(state["workspace"])).manifest()
    if not manifest:
        return ""
    listing = "\n".join(
        f"  {path}" + (f"  -> {', '.join(names)}" if names else "")
        for path, names in sorted(manifest.items())
    )
    return (
        f"\nTHE PROJECT ALREADY CONTAINS THESE FILES:\n{listing}\n\n"
        "Your plan is added to this existing project. Do not duplicate what is "
        "already there and do not rename it. If the TASK requires changing an "
        "existing file, list that same path in \"files\" and describe the change "
        "in its \"purpose\". Give this task's tests a test_file of their own, "
        "and names that will not collide with the tests already present.\n"
    )


def planner_node(state: AgencyState) -> dict[str, Any]:
    replans = state.get("replans", 0)
    previous = state.get("plan")
    review = state.get("review") or {}

    human = (state.get("plan_feedback") or "").strip()
    existing = _existing_project(state)

    if previous and human:
        # Your notes outrank the model's own opinion of its plan.
        user = (
            f"TASK:\n{state['task']}\n\n"
            f"You proposed this plan:\n"
            f"  summary: {previous.get('summary', '')}\n"
            f"  files: {', '.join(f['path'] for f in previous['files'])}\n"
            f"  tests:\n"
            + "\n".join(f"    - {_case_line(a)}" for a in previous["acceptance_tests"])
            + f"\n\nThe human reviewing it asked for these changes:\n{human}\n\n"
            "Produce a corrected plan that applies them. Treat the request as "
            "authoritative -- do not argue with it or partially apply it.\n\n"
            + SCHEMA
        )
    elif previous and review.get("test_file_defective"):
        user = (
            f"TASK:\n{state['task']}\n\n"
            f"Your previous plan produced a test suite the Reviewer judged defective.\n"
            f"Reviewer's reason: {review.get('defect_reason') or review.get('diagnosis')}\n\n"
            f"Previous acceptance tests:\n"
            + "\n".join(f"  - {_case_line(a)}" for a in previous["acceptance_tests"])
            + "\n\nProduce a corrected plan. Fix the faulty acceptance tests; keep the "
            "ones that were sound.\n\n" + SCHEMA
        )
        replans += 1
    else:
        user = f"TASK:\n{state['task']}\n{existing}\n{SCHEMA}"

    plan = llm.ask_json("planner", SYSTEM, user, validate=_validate)
    plan.setdefault("summary", "")
    plan.setdefault("notes", "")

    return {
        "plan": plan,
        "replans": replans,
        # A new plan invalidates the previous verdict and repair list.
        "review": {},
        "plan_feedback": "",  # consumed
        "events": [
            {
                "node": "planner",
                "replan": replans > state.get("replans", 0),
                "human_directed": bool(human),
                "summary": plan["summary"],
                "files": [f["path"] for f in plan["files"]],
                "test_file": plan["test_file"],
                "acceptance_tests": len(plan["acceptance_tests"]),
            }
        ],
    }
