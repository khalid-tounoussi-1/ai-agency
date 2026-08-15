"""The single state object every node reads and writes.

LangGraph merges each node's returned dict into this. Only `events` accumulates
(via the operator.add reducer); everything else is last-write-wins.
"""
import operator
from typing import Annotated, Any, TypedDict


class FileSpec(TypedDict):
    path: str
    purpose: str


class AcceptanceTest(TypedDict):
    name: str
    asserts: str


class Plan(TypedDict):
    summary: str
    files: list[FileSpec]
    test_file: str
    acceptance_tests: list[AcceptanceTest]
    notes: str


class TestReport(TypedDict):
    ok: bool
    stage: str  # "syntax" | "pytest" | "timeout"
    passed: int
    failed: int
    output: str


class Counterexample(TypedDict):
    name: str
    asserts: str
    why: str


class Review(TypedDict):
    verdict: str  # "APPROVE" | "REVISE"
    diagnosis: str
    files_to_fix: list[dict[str, str]]  # [{"path", "instruction"}]
    test_file_defective: bool
    defect_reason: str
    # Required to sustain a REVISE against a passing suite. See challenger.py.
    counterexample: Counterexample | None


class AgencyState(TypedDict, total=False):
    # Inputs
    task: str
    workspace: str

    # Produced by nodes
    plan: Plan
    files: dict[str, str]  # relative path -> content, as last written
    test_report: TestReport
    review: Review

    # Human gates
    gate_decision: str  # "approve"/"revise"/"abort" | "accept"/"changes"/"abort"
    plan_feedback: str  # your notes, consumed by the planner
    delivery_feedback: str  # your notes, consumed by the reviewer
    budget_bonus: int  # extra coder passes granted by a human request

    # Control
    iteration: int
    replans: int
    challenges: int
    challenge_rejected: str
    decision: str  # "repair" | "replan" | "done", written by the supervisor
    outcome: str  # "success" | "max_iterations" | "abandoned"
    events: Annotated[list[dict[str, Any]], operator.add]
