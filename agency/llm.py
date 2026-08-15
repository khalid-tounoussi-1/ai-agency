"""Model access, token accounting, and the JSON coercion the nodes rely on.

No tool calling anywhere in this project. The model returns text or JSON; every
side effect (writing a file, running a test) is performed by Python. On a
quantized local model that is the single largest reliability win available --
malformed tool calls were the dominant failure mode in the framework benchmark
this repo grew out of.
"""
import json
import re
import time
from typing import Any, Callable

from langchain_ollama import ChatOllama

from . import config


class Ledger:
    """Per-node token and wall-time accounting for one run."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, node: str, resp: Any, seconds: float) -> None:
        usage = getattr(resp, "usage_metadata", None) or {}
        self.calls.append(
            {
                "node": node,
                "seconds": round(seconds, 2),
                "prompt_tokens": usage.get("input_tokens") or 0,
                "completion_tokens": usage.get("output_tokens") or 0,
            }
        )

    def totals(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "seconds": round(sum(c["seconds"] for c in self.calls), 2),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "total_tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in self.calls),
        }

    def by_node(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            slot = out.setdefault(
                c["node"], {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            slot["calls"] += 1
            slot["seconds"] = round(slot["seconds"] + c["seconds"], 2)
            slot["prompt_tokens"] += c["prompt_tokens"]
            slot["completion_tokens"] += c["completion_tokens"]
        return out


LEDGER = Ledger()

_text_llm: ChatOllama | None = None
_json_llm: ChatOllama | None = None


def _build(json_mode: bool) -> ChatOllama:
    return ChatOllama(
        model=config.MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.NUM_CTX,
        # Ollama constrains sampling to valid JSON in this mode, which removes
        # most of the parse failures a quantized model would otherwise produce.
        format="json" if json_mode else None,
    )


def text_llm() -> ChatOllama:
    global _text_llm
    if _text_llm is None:
        _text_llm = _build(json_mode=False)
    return _text_llm


def json_llm() -> ChatOllama:
    global _json_llm
    if _json_llm is None:
        _json_llm = _build(json_mode=True)
    return _json_llm


def _invoke(llm: ChatOllama, node: str, system: str, user: str) -> str:
    started = time.perf_counter()
    resp = llm.invoke([("system", system), ("human", user)])
    LEDGER.record(node, resp, time.perf_counter() - started)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


_FENCE = re.compile(r"```[a-zA-Z0-9_+.-]*\n(.*?)(?:\n```|\Z)", re.S)


def strip_code_fences(text: str) -> str:
    """Recover raw source from a reply that may be fenced, prefaced with prose,
    or -- as observed in the benchmark -- terminated by a stray closing fence
    that was never opened."""
    blocks = [b for b in _FENCE.findall(text) if b.strip()]
    if blocks:
        body = max(blocks, key=len)
    else:
        body = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```"))
    return body.strip() + "\n"


def extract_json(text: str) -> str | None:
    """First balanced {...} object in the text, string-literal aware."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def ask_text(node: str, system: str, user: str) -> str:
    return _invoke(text_llm(), node, system, user)


def ask_json(
    node: str,
    system: str,
    user: str,
    validate: Callable[[dict], None] | None = None,
    attempts: int = 3,
) -> dict:
    """Ask for a JSON object, retrying with the parse/validation error fed back.

    `validate` should raise ValueError with a message the model can act on.
    """
    prompt = user
    last_error = ""
    for attempt in range(1, attempts + 1):
        raw = _invoke(json_llm(), node, system, prompt)
        blob = extract_json(raw)
        try:
            if blob is None:
                raise ValueError("no JSON object found in the reply")
            data = json.loads(blob)
            if not isinstance(data, dict):
                raise ValueError("top level value must be a JSON object")
            if validate is not None:
                validate(data)
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            prompt = (
                f"{user}\n\n"
                f"Your previous reply was rejected: {last_error}\n"
                "Reply with a single valid JSON object that fixes this. No prose."
            )
    raise RuntimeError(f"{node}: could not obtain valid JSON after {attempts} attempts: {last_error}")
