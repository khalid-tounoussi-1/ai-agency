"""Coder -- writes the test file first, then implements against it.

Two modes:

  build   (first pass, or after a replan) writes the pytest file from the
          plan's acceptance tests, then every source file, giving each one the
          frozen test file as its contract.
  repair  (later passes) rewrites only the files the Reviewer named, using the
          real pytest output as evidence. It is never given the option to edit
          the test file -- see the Reviewer for how a genuinely bad test gets
          fixed.
"""
from pathlib import Path
from typing import Any

from .. import llm
from ..state import AgencyState
from ..workspace import Workspace

SYSTEM = (
    "You are the Coder of a small software agency. You write one file at a "
    "time. You reply with the raw contents of that file and nothing else -- "
    "no explanation, no markdown fences."
)


def _plan_context(plan: dict[str, Any]) -> str:
    files = "\n".join(f"  {f['path']}  -- {f.get('purpose', '')}" for f in plan["files"])
    return (
        f"PLAN: {plan.get('summary', '')}\n"
        f"CONSTRAINTS: {plan.get('notes', '') or 'standard library only'}\n"
        f"SOURCE FILES IN THIS PROJECT:\n{files}\n"
    )


def _write_test_file(state: AgencyState, ws: Workspace, plan: dict[str, Any]) -> str:
    cases = "\n".join(
        f"  {a['name']}:\n    assert {a['asserts']}" for a in plan["acceptance_tests"]
    )
    user = (
        f"TASK:\n{state['task']}\n\n"
        f"{_plan_context(plan)}\n"
        f"You are writing exactly one file: {plan['test_file']}\n\n"
        f"Implement EXACTLY these acceptance tests as pytest functions, one "
        f"function per entry, using the given assertion verbatim where it is "
        f"valid Python:\n{cases}\n\n"
        "Import the functions under test from the source files listed above "
        "(they do not exist yet -- you are defining the contract they must "
        "meet). Do not weaken, skip, or wrap any assertion in a try block. "
        "Do not add mocks. Do not add tests beyond the list.\n\n"
        f"Reply with the raw contents of {plan['test_file']}."
    )
    content = llm.strip_code_fences(llm.ask_text("coder", SYSTEM, user))
    ws.write(plan["test_file"], content)
    return content


def _write_source_file(
    state: AgencyState,
    ws: Workspace,
    plan: dict[str, Any],
    spec: dict[str, str],
    test_source: str,
    written: dict[str, str],
) -> str:
    siblings = "".join(
        f"\n--- {path} (already written) ---\n{content}\n"
        for path, content in written.items()
        if path != spec["path"]
    )
    user = (
        f"TASK:\n{state['task']}\n\n"
        f"{_plan_context(plan)}\n"
        f"You are writing exactly one file: {spec['path']}\n"
        f"Its purpose: {spec.get('purpose', '')}\n\n"
        f"This test file is frozen and must pass unchanged. Match its imports "
        f"and its API exactly:\n--- {plan['test_file']} ---\n{test_source}\n"
        f"{siblings}\n"
        f"Reply with the raw contents of {spec['path']}."
    )
    content = llm.strip_code_fences(llm.ask_text("coder", SYSTEM, user))
    ws.write(spec["path"], content)
    return content


def _repair_file(
    state: AgencyState,
    ws: Workspace,
    plan: dict[str, Any],
    target: dict[str, str],
    files: dict[str, str],
) -> str:
    report = state.get("test_report") or {}
    review = state.get("review") or {}
    user = (
        f"TASK:\n{state['task']}\n\n"
        f"{_plan_context(plan)}\n"
        f"The build FAILED at the {report.get('stage', 'pytest')} stage.\n\n"
        f"Test runner output:\n{report.get('output', '')}\n\n"
        f"Reviewer's diagnosis: {review.get('diagnosis', '')}\n\n"
        f"Fix exactly one file: {target['path']}\n"
        f"What to change: {target.get('instruction', 'make the failing tests pass')}\n\n"
        f"Current contents of {target['path']}:\n{files.get(target['path'], '')}\n\n"
        f"The test file is frozen and may not be changed:\n"
        f"--- {plan['test_file']} ---\n{files.get(plan['test_file'], '')}\n\n"
        f"Reply with the complete corrected contents of {target['path']}. "
        "Not a diff, not a fragment -- the whole file."
    )
    content = llm.strip_code_fences(llm.ask_text("coder", SYSTEM, user))
    ws.write(target["path"], content)
    return content


def coder_node(state: AgencyState) -> dict[str, Any]:
    ws = Workspace(Path(state["workspace"]))
    plan = state["plan"]
    iteration = state.get("iteration", 0) + 1
    files = dict(state.get("files") or {})
    review = state.get("review") or {}
    repairs = review.get("files_to_fix") or []

    if not files or not repairs:
        mode = "build"
        files = {}
        test_source = _write_test_file(state, ws, plan)
        files[plan["test_file"]] = test_source
        for spec in plan["files"]:
            files[spec["path"]] = _write_source_file(state, ws, plan, spec, test_source, files)
        touched = list(files)
    else:
        mode = "repair"
        touched = []
        for target in repairs:
            files[target["path"]] = _repair_file(state, ws, plan, target, files)
            touched.append(target["path"])

    return {
        "files": files,
        "iteration": iteration,
        "events": [{"node": "coder", "iteration": iteration, "mode": mode, "wrote": touched}],
    }
