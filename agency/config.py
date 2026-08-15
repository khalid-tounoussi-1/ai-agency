"""Run configuration.

Every value is overridable by environment variable so a run can be pointed at a
different model or budget without editing code.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 32K Ollama variant. The raw upstream tag works too -- NUM_CTX below is
# sent explicitly on every request -- but the variant makes the default safe
# even for callers that forget.
MODEL = os.environ.get("AGENCY_MODEL", "qwen3coder-30b-32k")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama serves at num_ctx=4096 unless told otherwise. Agent prompts overflow
# that and the model degrades silently rather than erroring, so it is pinned
# per request here and never left to the server default.
NUM_CTX = int(os.environ.get("AGENCY_NUM_CTX", "32768"))
TEMPERATURE = float(os.environ.get("AGENCY_TEMPERATURE", "0"))

# A build attempt is one coder pass. Iteration 1 is the initial build; the rest
# are repairs driven by real test failures.
MAX_ITERATIONS = int(os.environ.get("AGENCY_MAX_ITERATIONS", "4"))

# How many times the Reviewer may send a defective plan back to the Planner.
MAX_REPLANS = int(os.environ.get("AGENCY_MAX_REPLANS", "1"))

# How many times the Reviewer may object to a green suite and have its
# counterexample executed. Beyond this, an unsubstantiated objection ends the
# run rather than spending the remaining budget on it.
MAX_CHALLENGES = int(os.environ.get("AGENCY_MAX_CHALLENGES", "2"))

# Generated code runs on this machine. Bound it.
TEST_TIMEOUT = int(os.environ.get("AGENCY_TEST_TIMEOUT", "120"))

RUNS_DIR = Path(os.environ.get("AGENCY_RUNS_DIR", REPO_ROOT / "runs"))
