"""The agency graph, and the supervisor that steers it.

The supervisor is a plain function over state. It holds no model, sends no
tokens, and can only return one of four decisions. Every route the agency can
take is visible in `supervisor_node` below -- which is the point of building it
this way. An LLM supervisor would put that logic in a prompt, where it cannot
be read, tested, or bounded.

    START -> planner -> [plan gate] -> coder -> tester -> reviewer -> supervisor
                ^            |           ^        ^                       |
                |          revise        |        |                       |
                +------------+           |        |                       |
                |                        +--------|----- repair ----------+
                +-------- replan --------|--------|-----------------------+
                                         |        |                       |
                                    challenger <--|------ challenge ------+
                                                  |                       |
                                        [delivery gate] <----- done ------+
                                                  |
                                    accept -> END | changes -> reviewer

The two gates in brackets are where a human accepts or rejects. With --review
off they are pass-throughs. Bounds, all in config: MAX_ITERATIONS coder passes
(plus HUMAN_BUDGET_BONUS per human request), MAX_REPLANS returns to the
planner, MAX_CHALLENGES executed reviewer objections.
"""
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import config
from .nodes.challenger import challenger_node
from .nodes.coder import coder_node
from .nodes.gate import delivery_gate_node, plan_gate_node
from .nodes.planner import planner_node
from .nodes.reviewer import reviewer_node
from .nodes.tester import tester_node
from .state import AgencyState


def supervisor_node(state: AgencyState) -> dict[str, Any]:
    review = state.get("review") or {}
    report = state.get("test_report") or {}
    iteration = state.get("iteration", 0)
    replans = state.get("replans", 0)
    challenges = state.get("challenges", 0)
    green = bool(report.get("ok"))
    budget = config.MAX_ITERATIONS + state.get("budget_bonus", 0)

    if review.get("human_directed") and review.get("verdict") == "REVISE":
        # You asked for this change. It does not need a counterexample to
        # justify it, and it is not subject to the reviewer's own judgement.
        decision, outcome, why = "repair", "", "human requested changes"
    elif green and review.get("verdict") == "APPROVE":
        decision, outcome, why = "done", "success", "tests green and reviewer approved"
    elif green and review.get("counterexample") and challenges < config.MAX_CHALLENGES:
        decision, outcome = "challenge", ""
        why = "reviewer objects to a green suite; executing its counterexample"
    elif green:
        # It wants changes but produced no failing case, or has spent its
        # challenges. Every acceptance test passes; that is the bar.
        decision, outcome = "done", "success"
        why = "reviewer objection unsubstantiated against a green suite"
    elif (
        report.get("syntax_file")
        and report.get("syntax_file") == (state.get("plan") or {}).get("test_file")
        and replans < config.MAX_REPLANS
    ):
        # The break is in the frozen test file, which the Coder may not touch.
        # Sending it to repair source code is an unfixable loop.
        decision, outcome, why = "replan", "", "the test file itself does not compile"
    elif report.get("stage") == "collection" and replans < config.MAX_REPLANS:
        # Nothing ran at all. Rewriting the source cannot fix that; the test
        # file itself is unusable, so the plan is redone.
        decision, outcome, why = "replan", "", "pytest collected no tests"
    elif review.get("test_file_defective") and replans < config.MAX_REPLANS:
        decision, outcome, why = "replan", "", "reviewer flagged the test suite as defective"
    elif iteration >= budget:
        decision, outcome, why = "done", "max_iterations", (
            f"iteration budget of {budget} exhausted with tests red"
        )
    else:
        decision, outcome, why = "repair", "", "tests are red; another pass"

    return {
        "decision": decision,
        "outcome": outcome,
        "events": [
            {
                "node": "supervisor",
                "decision": decision,
                "iteration": iteration,
                "reason": why,
            }
        ],
    }


def build_graph():
    graph = StateGraph(AgencyState)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("challenger", challenger_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("plan_gate", plan_gate_node)
    graph.add_node("delivery_gate", delivery_gate_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_gate")
    graph.add_conditional_edges(
        "plan_gate",
        lambda state: state.get("gate_decision", "approve"),
        {"approve": "coder", "revise": "planner", "abort": END},
    )
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("challenger", "tester")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["decision"],
        {
            "repair": "coder",
            "replan": "planner",
            "challenge": "challenger",
            "done": "delivery_gate",
        },
    )
    graph.add_conditional_edges(
        "delivery_gate",
        lambda state: state.get("gate_decision", "accept"),
        {"accept": END, "changes": "reviewer", "abort": END},
    )
    # interrupt() needs somewhere to persist the paused run so it can resume.
    return graph.compile(checkpointer=InMemorySaver())
