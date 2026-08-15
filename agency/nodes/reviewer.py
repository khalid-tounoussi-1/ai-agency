"""Reviewer -- reads the code alongside the real test output and decides.

Its verdict is advisory in one direction only. A failing test suite is an
automatic REVISE no matter what the model says, enforced below rather than
asked for in the prompt. When tests pass, the Reviewer's job is to catch the
implementation that satisfies the letter of the suite while missing the task.

The Reviewer may not hand the Coder the test file. If it believes a test is
itself wrong, it says so, and the run goes back to the Planner instead -- the
only route by which a test can change, and a bounded one.
"""
from typing import Any

from .. import llm
from ..state import AgencyState

SYSTEM = (
    "You are the Reviewer of a small software agency. You are shown code and "
    "the real output of its test run. You reply with a single JSON object."
)

SCHEMA = """Return a single JSON object with exactly these keys:

  "verdict"             "APPROVE" or "REVISE".
  "diagnosis"           what is wrong and why, in two or three sentences.
                        Cite the specific failing assertion or the specific
                        requirement that is unmet. Empty string if approving.
  "files_to_fix"        array of {"path", "instruction"}. One entry per source
                        file that must change. "instruction" is a concrete
                        change, not "fix the bug". Empty array if approving.
  "test_file_defective" true ONLY if a test itself is wrong -- it asserts
                        something the TASK does not require, or contradicts it.
                        A test that is merely hard to pass is NOT defective.
  "defect_reason"       which test is wrong and what it should assert instead.
                        Empty string unless test_file_defective is true.
  "counterexample"      null, or {"name", "asserts", "why"}."""

COUNTEREXAMPLE_RULE = """
Because every test currently passes, a REVISE verdict is only accepted if you
supply a "counterexample": a concrete input that the current code handles
wrongly, written as a new test case.

  "name"     a new pytest function name starting with "test_".
  "asserts"  a Python expression that SHOULD be true given the TASK, but is
             false for the code above. Use only names the test file already
             imports.
  "why"      one line: which requirement this input violates.

It will be appended to the suite and executed. If it passes, your objection is
recorded as unfounded. Do not guess -- pick an input you have traced through
the code by hand. If you cannot name one, the correct verdict is APPROVE."""


def _validate(data: dict[str, Any]) -> None:
    if str(data.get("verdict", "")).upper() not in {"APPROVE", "REVISE"}:
        raise ValueError('"verdict" must be exactly "APPROVE" or "REVISE"')
    fixes = data.get("files_to_fix", [])
    if not isinstance(fixes, list):
        raise ValueError('"files_to_fix" must be an array')
    for f in fixes:
        if not isinstance(f, dict) or not f.get("path") or not f.get("instruction"):
            raise ValueError('every entry in "files_to_fix" needs "path" and "instruction"')


def reviewer_node(state: AgencyState) -> dict[str, Any]:
    plan = state["plan"]
    files = state.get("files") or {}
    report = state.get("test_report") or {}
    source_paths = [f["path"] for f in plan["files"]]

    listing = "".join(
        f"\n--- {path} ---\n{files.get(path, '(missing)')}\n" for path in source_paths
    )
    test_listing = f"\n--- {plan['test_file']} ---\n{files.get(plan['test_file'], '')}\n"

    human = (state.get("delivery_feedback") or "").strip()

    if human:
        situation = (
            f"The suite passes, but the human who commissioned this work "
            f"rejected the delivery and asked for a change:\n\n{human}\n\n"
            "Their request is authoritative -- do not argue with it, do not "
            "judge whether it is necessary. Return REVISE and translate it "
            "into concrete per-file instructions the Coder can apply. Leave "
            "\"counterexample\" null."
        )
    elif report.get("ok"):
        situation = (
            f"All {report.get('passed', 0)} tests PASSED.\n\n"
            "Your job now is to decide whether this implementation genuinely "
            "satisfies the TASK, or whether it passes because the suite is "
            "narrower than the task. Do not approve code that special-cases "
            "the test inputs. Do not withhold approval over style, naming, "
            "type hints, or docstrings."
        )
        rejected = state.get("challenge_rejected")
        if rejected:
            situation += (
                f"\n\nYour previous counterexample was discarded: {rejected}. "
                "Supply a usable one or approve."
            )
    else:
        situation = (
            f"The build FAILED at the {report.get('stage', 'pytest')} stage "
            f"({report.get('passed', 0)} passed, {report.get('failed', 0)} failed).\n\n"
            "Your job is to diagnose the failure from the output below and say "
            "exactly which source file must change and how."
        )

    user = (
        f"TASK:\n{state['task']}\n\n"
        f"PLAN: {plan.get('summary', '')}\n"
        f"CONSTRAINTS: {plan.get('notes', '')}\n\n"
        f"{situation}\n\n"
        f"Test runner output:\n{report.get('output', '')}\n"
        f"{test_listing}{listing}\n"
        f"{SCHEMA}"
        f"{COUNTEREXAMPLE_RULE if report.get('ok') and not human else ''}"
    )

    review = llm.ask_json("reviewer", SYSTEM, user, validate=_validate)

    verdict = str(review.get("verdict", "REVISE")).upper()
    defective = bool(review.get("test_file_defective"))

    # Enforcement, not persuasion: a red suite cannot be approved, and neither
    # can a delivery a human has just rejected -- whatever the model returned.
    if not report.get("ok") or human:
        verdict = "REVISE"

    # The Coder never receives the test file. Anything outside the plan's source
    # list is dropped too -- the Reviewer does not get to invent new files.
    allowed = set(source_paths)
    fixes = [f for f in review.get("files_to_fix", []) if f.get("path") in allowed]
    if verdict == "REVISE" and not fixes and not defective:
        # It wants changes but named nothing actionable; fall back to the file
        # most likely at fault rather than looping on an empty repair list.
        fixes = [
            {
                "path": source_paths[0],
                "instruction": review.get("diagnosis") or "make the failing tests pass",
            }
        ]

    counter = review.get("counterexample")
    if not isinstance(counter, dict) or not counter.get("name") or not counter.get("asserts"):
        counter = None

    resolved = {
        "verdict": verdict,
        "diagnosis": str(review.get("diagnosis", "")),
        "files_to_fix": fixes,
        "test_file_defective": defective,
        "defect_reason": str(review.get("defect_reason", "")),
        "counterexample": None if human else counter,
        "human_directed": bool(human),
    }
    return {
        "review": resolved,
        "delivery_feedback": "",  # consumed
        "events": [
            {
                "node": "reviewer",
                "verdict": verdict,
                "human_directed": bool(human),
                "test_file_defective": defective,
                "files_to_fix": [f["path"] for f in fixes],
                "diagnosis": resolved["diagnosis"],
                "counterexample": counter["name"] if counter else None,
            }
        ],
    }
