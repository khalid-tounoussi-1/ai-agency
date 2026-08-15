"""Human gates -- the two points where the agency stops and asks you.

  plan gate      after the Planner, before a single line is written. This is
                 the cheapest place to intervene: rejecting a plan costs one
                 model call, rejecting a finished build costs the whole run.
  delivery gate  after the suite is green, before the run is called done.

Both use LangGraph's `interrupt()`, so the graph genuinely suspends and is
resumed with your answer rather than blocking inside a node on `input()`. The
terminal I/O lives in the CLI; this module only states what it needs answered.

With `--review` off, both gates are pass-throughs and the agency runs
unattended exactly as before.
"""
from typing import Any

from langgraph.types import interrupt

from .. import config
from ..state import AgencyState


def plan_gate_node(state: AgencyState) -> dict[str, Any]:
    if not config.INTERACTIVE:
        return {"gate_decision": "approve"}

    plan = state["plan"]
    answer = interrupt(
        {
            "gate": "plan",
            "summary": plan.get("summary", ""),
            "files": [f["path"] for f in plan["files"]],
            "test_file": plan["test_file"],
            "acceptance_tests": plan["acceptance_tests"],
            "notes": plan.get("notes", ""),
        }
    ) or {}

    action = str(answer.get("action", "approve")).lower()
    feedback = str(answer.get("feedback", "")).strip()

    if action == "revise" and not feedback:
        # Nothing to act on; treat it as approval rather than looping the
        # planner on no information.
        action = "approve"

    return {
        "gate_decision": action,
        "plan_feedback": feedback if action == "revise" else "",
        "outcome": "aborted_by_user" if action == "abort" else "",
        "events": [{"node": "gate", "gate": "plan", "action": action, "feedback": feedback}],
    }


def delivery_gate_node(state: AgencyState) -> dict[str, Any]:
    if not config.INTERACTIVE:
        return {"gate_decision": "accept"}

    report = state.get("test_report") or {}
    plan = state["plan"]
    answer = interrupt(
        {
            "gate": "delivery",
            "outcome": state.get("outcome", ""),
            "passed": report.get("passed", 0),
            "failed": report.get("failed", 0),
            "files": {path: (state.get("files") or {}).get(path, "") for path in
                      [f["path"] for f in plan["files"]] + [plan["test_file"]]},
        }
    ) or {}

    action = str(answer.get("action", "accept")).lower()
    feedback = str(answer.get("feedback", "")).strip()

    if action == "changes" and not feedback:
        action = "accept"

    update: dict[str, Any] = {
        "gate_decision": action,
        "events": [{"node": "gate", "gate": "delivery", "action": action, "feedback": feedback}],
    }
    if action == "changes":
        update["delivery_feedback"] = feedback
        update["outcome"] = ""
        # A change you asked for should not be refused because the automatic
        # budget happens to be spent.
        update["budget_bonus"] = state.get("budget_bonus", 0) + config.HUMAN_BUDGET_BONUS
    elif action == "abort":
        update["outcome"] = "aborted_by_user"
    return update
