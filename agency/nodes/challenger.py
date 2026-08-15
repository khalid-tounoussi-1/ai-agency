"""Challenger -- turns a Reviewer objection into an executable test.

When the suite is green and the Reviewer still objects, its claim is a
hypothesis, not a verdict. This node appends the counterexample it supplied to
the suite as a real pytest function and hands control back to the Tester. The
claim is then settled the same way everything else here is settled: by running
it.

  new test fails -> the Reviewer was right, and the suite is permanently
                    stronger for it.
  new test passes -> the objection was noise, and the run ends green.

There is no model call in this node. The test is assembled from the JSON the
Reviewer already returned.
"""
import ast
import builtins
import re
from pathlib import Path
from typing import Any

from ..state import AgencyState
from ..workspace import Workspace

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _known_names(test_source: str) -> set[str]:
    """Everything the existing test module can legally refer to."""
    names = set(dir(builtins))
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _vet(expression: str, test_source: str) -> str | None:
    """Reject a counterexample that would fail for its own reasons rather than
    the code's -- a syntax error, or a reference to something that does not
    exist. Returns a rejection reason, or None if the expression is sound."""
    try:
        tree = ast.parse(f"assert {expression}")
    except SyntaxError as exc:
        return f"not valid Python: {exc.msg}"

    known = _known_names(test_source)
    unknown = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known
    }
    if unknown:
        return f"refers to names the test module does not define: {', '.join(sorted(unknown))}"
    return None


def challenger_node(state: AgencyState) -> dict[str, Any]:
    plan = state["plan"]
    files = dict(state.get("files") or {})
    review = state.get("review") or {}
    counter = review.get("counterexample") or {}
    challenges = state.get("challenges", 0) + 1

    test_path = plan["test_file"]
    test_source = files.get(test_path, "")

    name = str(counter.get("name", "")).strip()
    expression = str(counter.get("asserts", "")).strip()
    expression = re.sub(r"^assert\s+", "", expression)

    if not IDENT.match(name) or not name.startswith("test_"):
        rejection = f"unusable test name {name!r}"
    elif not expression:
        rejection = "no assertion supplied"
    else:
        rejection = _vet(expression, test_source)

    if rejection:
        return {
            "challenges": challenges,
            "challenge_rejected": rejection,
            "events": [
                {"node": "challenger", "accepted": False, "reason": rejection, "name": name}
            ],
        }

    # Names in the suite must stay unique or pytest silently runs only the last.
    final_name = name
    suffix = 2
    while re.search(rf"^def {re.escape(final_name)}\(", test_source, re.M):
        final_name = f"{name}_{suffix}"
        suffix += 1

    why = str(counter.get("why", "")).strip()
    addition = (
        f"\n\n# added by the Reviewer as a counterexample"
        f"{': ' + why if why else ''}\n"
        f"def {final_name}():\n"
        f"    assert {expression}\n"
    )
    updated = test_source.rstrip("\n") + addition
    files[test_path] = updated

    ws = Workspace(Path(state["workspace"]))
    ws.write(test_path, updated)

    plan = dict(plan)
    plan["acceptance_tests"] = list(plan["acceptance_tests"]) + [
        {"name": final_name, "asserts": expression}
    ]

    return {
        "plan": plan,
        "files": files,
        "challenges": challenges,
        "challenge_rejected": "",
        "events": [
            {"node": "challenger", "accepted": True, "name": final_name, "asserts": expression}
        ],
    }
