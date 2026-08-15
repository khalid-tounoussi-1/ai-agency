"""Coder -- implements against a test file it did not write and cannot change.

Two modes:

  build   (first pass, or after a replan) renders the pytest file from the
          plan's acceptance tests -- mechanically, no model call, see
          testgen -- then writes every source file with that frozen file as
          its contract.
  repair  (later passes) rewrites only the files the Reviewer named, using the
          real pytest output as evidence. It is never given the option to edit
          the test file -- see the Reviewer for how a genuinely bad test gets
          fixed.
"""
from pathlib import Path
from typing import Any

from .. import llm, testgen
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


def _write_test_file(ws: Workspace, plan: dict[str, Any]) -> str:
    """Rendered from the plan, not written by the model. See testgen."""
    content = testgen.render(plan)
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
    # Working inside an existing project: this file may already exist, in which
    # case the task is to update it rather than replace it wholesale.
    current = ""
    if ws.exists(spec["path"]):
        current = (
            f"\nThis file ALREADY EXISTS with the contents below. Update it: keep "
            f"everything that still works and is used elsewhere, and do not drop "
            f"functions the task did not ask you to remove.\n"
            f"--- {spec['path']} (current) ---\n{ws.read(spec['path'])}\n"
        )
    user = (
        f"TASK:\n{state['task']}\n\n"
        f"{_plan_context(plan)}\n"
        f"You are writing exactly one file: {spec['path']}\n"
        f"Its purpose: {spec.get('purpose', '')}\n"
        f"{current}\n"
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
        test_source = _write_test_file(ws, plan)
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
