"""CLI entry point: `python -m agency "build me a thing"`."""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command

from . import config, llm, testgen
from .graph import build_graph
from .workspace import Workspace

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"
CYAN = "\033[36m"
RULE = "─" * 60


def _read(prompt: str, default: str = "") -> str:
    """Read one answer. A closed stdin means unattended, so take the default
    rather than crashing a piped run."""
    try:
        return input(prompt).strip()
    except EOFError:
        print(f"{DIM}(no input available, taking default){RESET}")
        return default


def ask_plan_gate(payload: dict[str, Any]) -> dict[str, str]:
    print(f"\n{CYAN}{RULE}\n  PLAN GATE -- nothing has been written yet{RESET}")
    print(f"  {BOLD}{payload['summary']}{RESET}")
    print(f"  files      {', '.join(payload['files'])}")
    print(f"  test file  {payload['test_file']}")
    if payload.get("notes"):
        print(f"  notes      {payload['notes']}")
    print(f"  {BOLD}acceptance tests{RESET} ({len(payload['acceptance_tests'])}) -- the contract "
          f"the code must satisfy:")
    for i, case in enumerate(payload["acceptance_tests"], 1):
        print(f"    {i:>2}. {case['name']}\n        {DIM}{testgen.describe(case)}{RESET}")
    print(f"{CYAN}{RULE}{RESET}")

    while True:
        choice = _read(
            f"  [{BOLD}a{RESET}]ccept   [{BOLD}r{RESET}]eject with notes   "
            f"[{BOLD}q{RESET}]uit  > ",
            "a",
        ).lower()
        if choice in {"", "a", "accept", "approve", "y"}:
            return {"action": "approve"}
        if choice in {"r", "reject", "revise", "n"}:
            notes = _read(f"  what should change? {DIM}(one line){RESET} > ")
            if notes:
                return {"action": "revise", "feedback": notes}
            print(f"  {YELLOW}no notes given -- nothing for the planner to act on{RESET}")
            continue
        if choice in {"q", "quit", "abort"}:
            return {"action": "abort"}
        print(f"  {YELLOW}answer a, r or q{RESET}")


def ask_delivery_gate(payload: dict[str, Any]) -> dict[str, str]:
    print(f"\n{CYAN}{RULE}\n  DELIVERY GATE -- the suite is green{RESET}")
    print(f"  tests   {GREEN}{payload['passed']} passed{RESET}, {payload['failed']} failed")
    print(f"  files   {', '.join(payload['files'])}")
    print(f"{CYAN}{RULE}{RESET}")

    while True:
        choice = _read(
            f"  [{BOLD}a{RESET}]ccept   [{BOLD}c{RESET}]hanges   "
            f"[{BOLD}v{RESET}]iew code   [{BOLD}q{RESET}]uit  > ",
            "a",
        ).lower()
        if choice in {"", "a", "accept", "y"}:
            return {"action": "accept"}
        if choice in {"v", "view"}:
            for path, content in payload["files"].items():
                print(f"\n{DIM}--- {path} ---{RESET}\n{content}")
            continue
        if choice in {"c", "changes", "reject", "n"}:
            notes = _read(f"  what should change? {DIM}(one line){RESET} > ")
            if notes:
                return {"action": "changes", "feedback": notes}
            print(f"  {YELLOW}no notes given -- nothing to act on{RESET}")
            continue
        if choice in {"q", "quit", "abort"}:
            return {"action": "abort"}
        print(f"  {YELLOW}answer a, c, v or q{RESET}")


def slugify(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:5]
    return "-".join(words) or "task"


def preflight() -> None:
    """Confirm the model is actually installed before spending a minute on a
    run that will fail on the first call."""
    url = f"{config.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            tags = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        sys.exit(f"{RED}cannot reach Ollama at {config.OLLAMA_BASE_URL}{RESET} ({exc})")

    names = {m["name"] for m in tags.get("models", [])}
    if config.MODEL in names or f"{config.MODEL}:latest" in names:
        return
    listing = "\n".join(f"  {n}" for n in sorted(names))
    sys.exit(
        f"{RED}model {config.MODEL!r} is not installed.{RESET}\n"
        f"available:\n{listing}\n\n"
        f"set AGENCY_MODEL, or create the 32K variant:\n"
        f"  printf 'FROM hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q3_K_XL\\n"
        f"PARAMETER num_ctx 32768\\n' > Modelfile\n"
        f"  ollama create qwen3coder-30b-32k -f Modelfile"
    )


def git_dirty(root: Path) -> int:
    """How many tracked files have uncommitted changes, or 0 if this is not a
    git repo. Only used to warn before overwriting someone's work."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if proc.returncode != 0:
        return 0
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def render(event: dict[str, Any]) -> str:
    node = event["node"]
    if node == "planner":
        tag = "replan" if event.get("replan") else "plan"
        return (
            f"{BOLD}planner{RESET} [{tag}] {event['summary']}\n"
            f"  {DIM}files: {', '.join(event['files'])} | tests: {event['test_file']} "
            f"({event['acceptance_tests']} cases){RESET}"
        )
    if node == "coder":
        return (
            f"{BOLD}coder{RESET} [pass {event['iteration']}, {event['mode']}] "
            f"wrote {', '.join(event['wrote'])}"
        )
    if node == "tester":
        if event["stage"] != "pytest":
            return f"{BOLD}tester{RESET} {RED}{event['stage']} error{RESET}"
        colour = GREEN if event["ok"] else RED
        return (
            f"{BOLD}tester{RESET} {colour}{event['passed']} passed, "
            f"{event['failed']} failed{RESET}"
        )
    if node == "reviewer":
        colour = GREEN if event["verdict"] == "APPROVE" else YELLOW
        line = f"{BOLD}reviewer{RESET} {colour}{event['verdict']}{RESET}"
        if event.get("test_file_defective"):
            line += f" {YELLOW}(test suite flagged defective){RESET}"
        if event.get("diagnosis"):
            line += f"\n  {DIM}{event['diagnosis']}{RESET}"
        if event.get("counterexample"):
            line += f"\n  {DIM}counterexample: {event['counterexample']}{RESET}"
        if event.get("files_to_fix"):
            line += f"\n  {DIM}fix: {', '.join(event['files_to_fix'])}{RESET}"
        return line
    if node == "challenger":
        if event["accepted"]:
            return (
                f"{BOLD}challenger{RESET} appended {event['name']}\n"
                f"  {DIM}assert {event['asserts']}{RESET}"
            )
        return f"{BOLD}challenger{RESET} {YELLOW}rejected{RESET} {DIM}{event['reason']}{RESET}"
    if node == "supervisor":
        return f"{DIM}supervisor -> {event['decision']} ({event['reason']}){RESET}"
    if node == "gate":
        line = f"{CYAN}you {event['action']}ed the {event['gate']}{RESET}"
        if event.get("feedback"):
            line += f"\n  {DIM}{event['feedback']}{RESET}"
        return line
    return json.dumps(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agency", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("task", nargs="?", help="the task, as a sentence")
    source.add_argument("-f", "--file", type=Path, help="read the task from a file")
    parser.add_argument(
        "--into",
        type=Path,
        help="build into this project directory instead of a fresh run workspace",
    )
    parser.add_argument("--name", help="run directory name (default: slug of the task)")
    parser.add_argument("--model", help=f"override AGENCY_MODEL (default {config.MODEL})")
    parser.add_argument("--max-iterations", type=int, help="coder passes before giving up")
    parser.add_argument(
        "--review",
        action="store_true",
        help="stop for your approval at the plan and at the delivery",
    )
    args = parser.parse_args(argv)

    if args.model:
        config.MODEL = args.model
    if args.max_iterations:
        config.MAX_ITERATIONS = args.max_iterations
    if args.review:
        config.INTERACTIVE = True

    task = args.file.read_text().strip() if args.file else args.task.strip()
    if not task:
        return parser.error("the task is empty")

    preflight()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / f"{args.name or slugify(task)}-{stamp}"

    if args.into:
        # Working inside a project the user owns: the code lands there, while
        # the run's own record stays in runs/ rather than littering their repo.
        ws = Workspace(args.into.expanduser().resolve())
        dirty = git_dirty(ws.root)
        if dirty:
            print(
                f"{YELLOW}note{RESET} {ws.root} has {dirty} uncommitted change(s). "
                f"The agency overwrites files it is asked to write.\n"
            )
    else:
        ws = Workspace(run_dir / "workspace")

    print(f"{BOLD}task{RESET} {task}")
    print(f"{DIM}model {config.MODEL} | ctx {config.NUM_CTX} | workspace {ws.root}{RESET}")
    print(
        f"{DIM}{'stopping for your approval at the plan and the delivery' if config.INTERACTIVE else 'unattended -- pass --review to approve the plan before it builds'}{RESET}\n"
    )

    app = build_graph()
    run_config = {"recursion_limit": 100, "configurable": {"thread_id": run_dir.name}}
    inputs: Any = {
        "task": task,
        "workspace": str(ws.root),
        "iteration": 0,
        "replans": 0,
        "events": [],
    }

    started = time.perf_counter()
    final: dict[str, Any] = {}
    shown = 0
    error: str | None = None
    asked = {"plan": ask_plan_gate, "delivery": ask_delivery_gate}
    try:
        while True:
            for chunk in app.stream(inputs, stream_mode="values", config=run_config):
                final = chunk
                for event in chunk.get("events", [])[shown:]:
                    print(render(event))
                shown = len(chunk.get("events", []))

            # A gate suspended the graph. Ask, then resume it with the answer.
            snapshot = app.get_state(run_config)
            final = snapshot.values or final
            pending = [i for task_ in snapshot.tasks for i in task_.interrupts]
            if not pending:
                break
            payload = pending[0].value
            inputs = Command(resume=asked[payload["gate"]](payload))
    except KeyboardInterrupt:
        error = "interrupted"
        print(f"\n{YELLOW}stopped{RESET}")
    except Exception as exc:  # noqa: BLE001 - the run is over either way
        error = f"{exc.__class__.__name__}: {exc}"
        print(f"\n{RED}run aborted{RESET} {error}")

    elapsed = round(time.perf_counter() - started, 2)
    report = final.get("test_report") or {}
    outcome = final.get("outcome") or ("aborted" if error else "incomplete")
    totals = llm.LEDGER.totals()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in final.get("events", []))
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "task": task,
                "model": config.MODEL,
                "num_ctx": config.NUM_CTX,
                "outcome": outcome,
                "error": error,
                "elapsed_seconds": elapsed,
                "iterations": final.get("iteration", 0),
                "replans": final.get("replans", 0),
                "challenges": final.get("challenges", 0),
                "plan": final.get("plan"),
                "test_report": report,
                "review": final.get("review"),
                "files": sorted((final.get("files") or {}).keys()),
                "tokens": totals,
                "tokens_by_node": llm.LEDGER.by_node(),
            },
            indent=2,
        )
        + "\n"
    )

    colour = GREEN if outcome == "success" else RED
    print(
        f"\n{BOLD}outcome{RESET} {colour}{outcome}{RESET} | "
        f"{report.get('passed', 0)} passed, {report.get('failed', 0)} failed | "
        f"{final.get('iteration', 0)} coder pass(es) | {elapsed}s | "
        f"{totals['total_tokens']} tokens ({totals['calls']} calls)"
    )
    print(f"{DIM}{run_dir}{RESET}")
    return 0 if outcome == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
