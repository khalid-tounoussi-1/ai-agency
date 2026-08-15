"""Renders the Planner's acceptance tests into a pytest file.

This is deliberately mechanical. The Planner supplies the *content* of each
test as structured JSON; Python assembles the file. Nothing writes the test
file freehand.

That is not a stylistic preference. In the first version the Coder wrote the
test file from a prose description of each case, and the Planner handed it
`registry.add_student('', ...) raises ValueError` -- English, not Python. The
Coder faithfully rendered `assert <that>`, the file would not compile, and
because the test file is frozen the Coder could only rewrite the source: an
unfixable loop that burned every iteration.

Now every case is compiled here, at plan time, before anything is built. A case
that does not parse is rejected while it still costs one model call to fix.
"""
import ast
from typing import Any


def function_source(case: dict[str, Any]) -> str:
    """The pytest function for one acceptance test."""
    lines = [f"def {case['name']}():"]
    for statement in case.get("setup") or []:
        lines.append(f"    {statement}")
    if case.get("raises"):
        lines.append(f"    with pytest.raises({case['raises']}):")
        lines.append(f"        {case['call']}")
    else:
        lines.append(f"    assert {case['asserts']}")
    return "\n".join(lines)


def describe(case: dict[str, Any]) -> str:
    """One readable line for a case, in either form. Used wherever a plan is
    shown to a human or fed back to the Planner."""
    setup = "; ".join(case.get("setup") or [])
    prefix = f"{setup}  ->  " if setup else ""
    if case.get("raises"):
        return f"{prefix}{case.get('call', '')} raises {case['raises']}"
    return f"{prefix}assert {case.get('asserts', '')}"


def validate_case(case: dict[str, Any]) -> None:
    """Raise ValueError with a message the Planner can act on."""
    name = str(case.get("name", ""))
    if not name.startswith("test_") or not name.isidentifier():
        raise ValueError(f"test name must be a Python identifier starting with 'test_': {name!r}")

    has_assert = bool(str(case.get("asserts", "")).strip())
    has_raises = bool(str(case.get("raises", "")).strip())
    if has_assert == has_raises:
        raise ValueError(
            f"{name}: give exactly one of \"asserts\" (an expression) or "
            f'"raises" plus "call" (an exception name and the expression that raises it)'
        )
    if has_raises and not str(case.get("call", "")).strip():
        raise ValueError(f'{name}: "raises" also needs "call", the expression that raises it')

    setup = case.get("setup") or []
    if not isinstance(setup, list) or any(not isinstance(s, str) for s in setup):
        raise ValueError(f'{name}: "setup" must be an array of Python statements')

    try:
        ast.parse(function_source(case))
    except SyntaxError as exc:
        raise ValueError(
            f"{name}: this does not compile as Python ({exc.msg} on line {exc.lineno}). "
            f"Write real Python -- 'f(x) raises ValueError' is English, not an expression; "
            f'use "raises" and "call" for that.'
        ) from exc


def validate_imports(imports: Any) -> None:
    if not isinstance(imports, list) or not imports:
        raise ValueError('"test_imports" must be a non-empty array of Python import lines')
    for line in imports:
        try:
            tree = ast.parse(str(line))
        except SyntaxError as exc:
            raise ValueError(f"not a valid import line: {line!r} ({exc.msg})") from exc
        if not tree.body or not all(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
        ):
            raise ValueError(f"only import statements belong in test_imports: {line!r}")


def render(plan: dict[str, Any]) -> str:
    """The complete pytest file for a plan."""
    header = ["import pytest", ""] + list(plan.get("test_imports") or [])
    functions = "\n\n\n".join(function_source(c) for c in plan["acceptance_tests"])
    return "\n".join(header).rstrip("\n") + "\n\n\n" + functions + "\n"
