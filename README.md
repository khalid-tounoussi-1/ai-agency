# ai-agency

A small software agency built on LangGraph, driven entirely by a local model.
You give it a spec; a Planner turns it into acceptance tests, a Coder writes
the code, a Tester **executes** it, and a Reviewer reads the real failures. It
loops until the suite is green or the budget runs out.

Nothing is judged by an LLM. A run succeeds when assertions pass.

Built on top of the framework comparison in [`ai-benchmarks/multiagent`](../ai-benchmarks/multiagent),
which is why LangGraph is the substrate: it was the leanest of the six and the
only one where every prompt is written by hand rather than implied.

## Quick start

```sh
cd ~/Desktop/src/repos/ai-agency
uv venv --python 3.12
uv pip install langgraph langchain-ollama pytest
```

You also need the 32K model variant (see [The context requirement](#the-context-requirement)):

```sh
printf 'FROM hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q3_K_XL\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen3coder-30b-32k -f Modelfile
```

Then commission something:

```sh
# a spec in a file -- the normal way
.venv/bin/python -m agency -f specs/school-management/01-student-registry.md

# a one-liner, for something small
.venv/bin/python -m agency "a function that parses ISO durations into seconds"

# stop and ask you before writing code, and again before finishing
.venv/bin/python -m agency -f myspec.md --review
```

Exit status is 0 when the suite is green, 1 otherwise, so it drops into a
script or a Makefile without parsing the output.

### Options

| Flag | Effect |
|---|---|
| `-f, --file PATH` | Read the spec from a file instead of the command line |
| `--review` | Stop at the plan gate and the delivery gate for your approval |
| `--name NAME` | Name the run directory (default: a slug of the spec) |
| `--model NAME` | Use a different Ollama model for this run |
| `--max-iterations N` | Coder passes before giving up (default 4) |

Anything you would set repeatedly is an environment variable instead:
`AGENCY_MODEL`, `AGENCY_NUM_CTX`, `AGENCY_MAX_ITERATIONS`, `AGENCY_MAX_REPLANS`,
`AGENCY_MAX_CHALLENGES`, `AGENCY_TEST_TIMEOUT`, `AGENCY_TEMPERATURE`,
`AGENCY_RUNS_DIR`, `OLLAMA_BASE_URL`.

## Where the work lands

Every run gets its own directory under `runs/`:

```
runs/<name>-<timestamp>/
├── workspace/          the project, exactly as the agency left it
│   ├── <source>.py
│   ├── tests/test_<x>.py
│   └── pytest.ini      pins pytest's config to this directory
├── run.json            plan, outcome, test report, tokens per node
└── events.jsonl        every decision in order, one line each
```

The workspace is a real project. `cd` into it and run `pytest` yourself.

`run.json` is the artifact worth keeping — it records the plan, the final test
report, and the token and wall-time cost broken down per node, which is what
makes runs comparable to each other.

## Approving and rejecting

With `--review`, the agency stops twice.

**The plan gate**, after planning and before a single line is written. You see
the files it intends to create and every acceptance test it intends to hold
itself to:

```
────────────────────────────────────────────────────────────
  PLAN GATE -- nothing has been written yet
  A student registry with validation and lookup by grade.
  files      student_registry.py
  test file  tests/test_student_registry.py
  acceptance tests (9) -- the contract the code must satisfy:
     1. test_add_student_returns_record
        assert StudentRegistry().add_student("s1", "Ada", 9, "a@x.com")["active"] is True
     ...
────────────────────────────────────────────────────────────
  [a]ccept   [r]eject with notes   [q]uit  >
```

Rejecting sends it back to the Planner with your notes, which are treated as
authoritative. This is the cheap place to intervene: rejecting a plan costs one
model call, rejecting a finished build costs the entire run.

**The delivery gate**, once the suite is green. You can `[v]iew` the code
before deciding, and asking for changes routes your note through the Reviewer,
which turns it into per-file instructions for the Coder — and grants extra
iterations so your request is not refused by an exhausted budget.

Without `--review` both gates are pass-throughs and the run is unattended.

## Writing a spec

The Planner converts your spec into pytest assertions, one per requirement. A
requirement it cannot turn into an assertion is a requirement that silently
does not get built. So write down:

- exact names and argument order for every function and class
- the exact shape of return values, key by key
- the exception type for each bad input, **by name**
- the boundaries: empty, missing, duplicate, zero, out of range

"Validate the input and handle errors gracefully" produces a vague test and
code that passes it without doing anything. "`grade_level` must be an `int`
from 1 to 12 inclusive, otherwise `ValueError`" produces a test that bites.

[`specs/school-management/`](specs/school-management/) is five worked examples.

## How it works

```
START -> planner -> [plan gate] -> coder -> tester -> reviewer -> supervisor
            ^            |           ^        ^                       |
            |          reject        |        |                       |
            +------------+           |        |                       |
            |                        +--------|----- repair ----------+
            +-------- replan --------|--------|-----------------------+
                                     |        |                       |
                                challenger <--|------ challenge ------+
                                              |                       |
                                    [delivery gate] <----- done ------+
                                              |
                                accept -> END | changes -> reviewer
```

**Planner** turns the spec into a file layout and a list of concrete acceptance
tests. The tests are written here, from the spec, before any implementation
exists — so the contract comes from the requirement rather than from whatever
code happened to get written.

**Coder** writes the test file first, then implements each source file against
it. It is never allowed to edit the test file afterwards. An agent that can
edit its own tests will eventually edit them instead of fixing the bug.

**Tester** has no model in it. It compiles every file, then runs pytest in a
subprocess and reports what happened.

**Reviewer** reads the code next to the *real* pytest output. A red suite is an
automatic REVISE regardless of what it says — enforced in code, not requested
in the prompt.

**Supervisor** is a plain function over state. It holds no model and sends no
tokens; every route the agency can take is [thirty readable lines](agency/graph.py).

### The counterexample protocol

The first working version had a flaw worth describing, because it is the
central problem with LLM review at this model size.

On the very first real task, the suite went green on pass 1 — and the Reviewer
returned REVISE anyway, with a confidently-worded diagnosis that was simply
wrong. It repeated it three times, word for word, until the iteration budget
was gone. The code had been correct for 60 seconds.

A model that objects for free will object forever. So an objection now costs
something:

> When the suite is green, a REVISE is only accepted with a **counterexample** —
> a concrete input, written as an assertion, that the current code gets wrong.

The Challenger appends it to the suite as a real test and the Tester runs it.
If it fails, the Reviewer was right and the suite is permanently stronger. If
it passes, the objection was noise and the run ends green. If no counterexample
is offered, the objection is discarded.

The Challenger vets the assertion first, by AST, rejecting anything that
references a name the test module does not define — otherwise a bogus
counterexample fails with `NameError` and gets misread as a defect in the code.

Objections are bounded at `MAX_CHALLENGES` either way.

### No tool calls, anywhere

The model never calls a tool. It returns text or JSON; Python performs every
side effect — writing files, running pytest, appending tests. On a quantized
local model this is the single largest reliability win available. Malformed
tool calls were the dominant failure mode across the frameworks benchmarked
next door, and this design has no surface for them.

The same reasoning drives the deterministic supervisor. An LLM router would put
the control flow inside a prompt, where it cannot be read, tested, or bounded —
and would spend a model call per hop to get there.

## Tests

The agency's own logic is tested the way it tests its output — by executing it.
No Ollama call happens in the suite.

```sh
.venv/bin/python -m pytest
```

33 tests covering supervisor routing, challenger vetting, workspace path
safety, reply parsing, and the tester's verdicts.

## The context requirement

Ollama serves every model at `num_ctx=4096` unless told otherwise, and an
overflowing agent prompt does not error — the model quietly degrades. Every
request here sends `num_ctx` explicitly, and the default model is the 32K
variant, so this is handled twice over. It is still the first thing to check if
a local agent starts behaving strangely.

## Caveats

- **Generated code runs on this machine**, in a subprocess with no sandbox,
  bounded only by `AGENCY_TEST_TIMEOUT`. Read a spec before you run it if it
  came from somewhere else.
- Scope is one Python module plus its tests. It builds a library, not an app
  with a UI or a database — larger systems are commissioned one module at a
  time, which is what [`specs/school-management/`](specs/school-management/)
  demonstrates.
- Green means *the acceptance tests pass*. If the plan under-specifies the
  spec, green is still reachable with a gap in it. That is what the delivery
  gate and the counterexample protocol exist to narrow, and neither closes it.
- `temperature: 0`, but MoE routing keeps runs non-deterministic. The same spec
  can plan differently twice.
