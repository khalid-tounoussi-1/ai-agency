"""CLI entry point: `python -m agency "build me a thing"`."""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config, llm
from .graph import build_graph
from .workspace import Workspace

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"


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
    return json.dumps(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agency", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("task", nargs="?", help="the task, as a sentence")
    source.add_argument("-f", "--file", type=Path, help="read the task from a file")
    parser.add_argument("--name", help="run directory name (default: slug of the task)")
    parser.add_argument("--model", help=f"override AGENCY_MODEL (default {config.MODEL})")
    parser.add_argument("--max-iterations", type=int, help="coder passes before giving up")
    args = parser.parse_args(argv)

    if args.model:
        config.MODEL = args.model
    if args.max_iterations:
        config.MAX_ITERATIONS = args.max_iterations

    task = args.file.read_text().strip() if args.file else args.task.strip()
    if not task:
        return parser.error("the task is empty")

    preflight()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / f"{args.name or slugify(task)}-{stamp}"
    ws = Workspace(run_dir / "workspace")

    print(f"{BOLD}task{RESET} {task}")
    print(f"{DIM}model {config.MODEL} | ctx {config.NUM_CTX} | workspace {ws.root}{RESET}\n")

    app = build_graph()
    initial = {"task": task, "workspace": str(ws.root), "iteration": 0, "replans": 0, "events": []}

    started = time.perf_counter()
    final: dict[str, Any] = {}
    shown = 0
    error: str | None = None
    try:
        for chunk in app.stream(initial, stream_mode="values", config={"recursion_limit": 100}):
            final = chunk
            for event in chunk.get("events", [])[shown:]:
                print(render(event))
            shown = len(chunk.get("events", []))
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
