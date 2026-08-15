"""Tests for the parts of the agency that must not depend on a model.

The supervisor's routing, the challenger's vetting, the workspace's path
checks and the tester's verdict are all pure functions of state. They are
tested here the same way the agency tests its own output: by executing them.
No Ollama call happens in this file.
"""
import pytest

from agency import config, llm
from agency.graph import supervisor_node
from agency.nodes.challenger import _vet, challenger_node
# Imported as a module: pytest collects anything named test*, and importing
# `tester_node` directly would make it a test case.
from agency.nodes import tester
from agency.workspace import PathRejected, Workspace

GREEN = {"ok": True, "stage": "pytest", "passed": 5, "failed": 0, "output": ""}
RED = {"ok": False, "stage": "pytest", "passed": 3, "failed": 2, "output": "boom"}


def review(**kwargs):
    base = {
        "verdict": "REVISE",
        "diagnosis": "",
        "files_to_fix": [],
        "test_file_defective": False,
        "defect_reason": "",
        "counterexample": None,
    }
    return base | kwargs


class TestSupervisorRouting:
    def test_green_and_approved_is_success(self):
        out = supervisor_node({"test_report": GREEN, "review": review(verdict="APPROVE")})
        assert out["decision"] == "done"
        assert out["outcome"] == "success"

    def test_green_objection_with_counterexample_is_challenged(self):
        out = supervisor_node(
            {
                "test_report": GREEN,
                "review": review(counterexample={"name": "test_x", "asserts": "f(1) == 2"}),
                "challenges": 0,
            }
        )
        assert out["decision"] == "challenge"

    def test_green_objection_without_counterexample_ends_the_run(self):
        """The failure this protocol exists to prevent: a noisy REVISE against
        a passing suite spending the whole iteration budget."""
        out = supervisor_node({"test_report": GREEN, "review": review(diagnosis="feels wrong")})
        assert out["decision"] == "done"
        assert out["outcome"] == "success"

    def test_challenges_are_bounded(self):
        out = supervisor_node(
            {
                "test_report": GREEN,
                "review": review(counterexample={"name": "test_x", "asserts": "f(1) == 2"}),
                "challenges": config.MAX_CHALLENGES,
            }
        )
        assert out["decision"] == "done"

    def test_red_routes_to_repair(self):
        out = supervisor_node({"test_report": RED, "review": review(), "iteration": 1})
        assert out["decision"] == "repair"
        assert out["outcome"] == ""

    def test_red_at_budget_gives_up(self):
        out = supervisor_node(
            {"test_report": RED, "review": review(), "iteration": config.MAX_ITERATIONS}
        )
        assert out["decision"] == "done"
        assert out["outcome"] == "max_iterations"

    def test_defective_suite_routes_back_to_planner(self):
        out = supervisor_node(
            {
                "test_report": RED,
                "review": review(test_file_defective=True),
                "iteration": 1,
                "replans": 0,
            }
        )
        assert out["decision"] == "replan"

    def test_replans_are_bounded(self):
        out = supervisor_node(
            {
                "test_report": RED,
                "review": review(test_file_defective=True),
                "iteration": 1,
                "replans": config.MAX_REPLANS,
            }
        )
        assert out["decision"] == "repair"


class TestChallengerVetting:
    SOURCE = "import pytest\nfrom mod import flatten\n\ndef test_a():\n    assert flatten([]) == []\n"

    def test_sound_expression_is_accepted(self):
        assert _vet("flatten([1, [2]]) == [1, 2]", self.SOURCE) is None

    def test_syntax_error_is_rejected(self):
        assert "not valid Python" in _vet("flatten([1,) == 2", self.SOURCE)

    def test_unknown_name_is_rejected(self):
        """Guards against a counterexample that would fail with NameError and be
        misread as a defect in the code under test."""
        reason = _vet("unflatten([1]) == [1]", self.SOURCE)
        assert "unflatten" in reason

    def test_builtins_are_known(self):
        assert _vet("len(flatten([1, [2]])) == 2", self.SOURCE) is None


class TestChallengerNode:
    def _state(self, tmp_path, counterexample):
        ws = Workspace(tmp_path)
        source = "from mod import flatten\n\ndef test_a():\n    assert flatten([]) == []\n"
        ws.write("tests/test_mod.py", source)
        return {
            "workspace": str(tmp_path),
            "plan": {"test_file": "tests/test_mod.py", "acceptance_tests": [{"name": "test_a", "asserts": "flatten([]) == []"}]},
            "files": {"tests/test_mod.py": source},
            "review": {"counterexample": counterexample},
            "challenges": 0,
        }

    def test_accepted_counterexample_is_appended_to_the_suite(self, tmp_path):
        state = self._state(tmp_path, {"name": "test_deep", "asserts": "flatten([[[1]]]) == [1]", "why": "depth"})
        out = challenger_node(state)
        assert out["challenges"] == 1
        appended = out["files"]["tests/test_mod.py"]
        assert "def test_deep():" in appended
        assert "assert flatten([[[1]]]) == [1]" in appended
        # and it reached disk, so the tester will run it
        assert "def test_deep():" in (tmp_path / "tests/test_mod.py").read_text()
        assert len(out["plan"]["acceptance_tests"]) == 2

    def test_leading_assert_keyword_is_tolerated(self, tmp_path):
        state = self._state(tmp_path, {"name": "test_deep", "asserts": "assert flatten([]) == []"})
        out = challenger_node(state)
        assert "    assert flatten([]) == []" in out["files"]["tests/test_mod.py"]

    def test_duplicate_name_is_uniquified(self, tmp_path):
        state = self._state(tmp_path, {"name": "test_a", "asserts": "flatten([1]) == [1]"})
        out = challenger_node(state)
        assert "def test_a_2():" in out["files"]["tests/test_mod.py"]

    def test_bad_name_is_rejected_without_touching_the_suite(self, tmp_path):
        state = self._state(tmp_path, {"name": "not a name", "asserts": "flatten([]) == []"})
        out = challenger_node(state)
        assert out["challenge_rejected"]
        assert "files" not in out


class TestWorkspaceSafety:
    def test_traversal_is_refused(self, tmp_path):
        with pytest.raises(PathRejected):
            Workspace(tmp_path).write("../escaped.py", "x = 1")

    def test_absolute_path_is_refused(self, tmp_path):
        with pytest.raises(PathRejected):
            Workspace(tmp_path).write("/etc/passwd", "x = 1")

    def test_nested_write_creates_parents(self, tmp_path):
        Workspace(tmp_path).write("pkg/sub/mod.py", "x = 1")
        assert (tmp_path / "pkg/sub/mod.py").read_text() == "x = 1"


class TestTester:
    def test_syntax_error_short_circuits_before_pytest(self, tmp_path):
        out = tester.tester_node({"workspace": str(tmp_path), "files": {"mod.py": "def broken(:\n"}})
        assert out["test_report"]["ok"] is False
        assert out["test_report"]["stage"] == "syntax"

    def test_passing_suite_is_reported_green(self, tmp_path):
        files = {
            "mod.py": "def double(x):\n    return x * 2\n",
            "tests/test_mod.py": "from mod import double\n\ndef test_double():\n    assert double(2) == 4\n",
        }
        ws = Workspace(tmp_path)
        for path, content in files.items():
            ws.write(path, content)
        report = tester.tester_node({"workspace": str(tmp_path), "files": files})["test_report"]
        assert report["ok"] is True
        assert report["passed"] == 1

    def test_failing_suite_is_reported_red(self, tmp_path):
        files = {
            "mod.py": "def double(x):\n    return x * 3\n",
            "tests/test_mod.py": "from mod import double\n\ndef test_double():\n    assert double(2) == 4\n",
        }
        ws = Workspace(tmp_path)
        for path, content in files.items():
            ws.write(path, content)
        report = tester.tester_node({"workspace": str(tmp_path), "files": files})["test_report"]
        assert report["ok"] is False
        assert report["failed"] == 1

    def test_empty_suite_is_not_a_pass(self, tmp_path):
        """Deleting the tests must never look like success."""
        ws = Workspace(tmp_path)
        ws.write("mod.py", "x = 1\n")
        report = tester.tester_node({"workspace": str(tmp_path), "files": {"mod.py": "x = 1\n"}})[
            "test_report"
        ]
        assert report["ok"] is False


class TestReplyParsing:
    def test_fenced_block_is_unwrapped(self):
        assert llm.strip_code_fences("Here you go:\n```python\nx = 1\n```") == "x = 1\n"

    def test_bare_code_survives(self):
        assert llm.strip_code_fences("x = 1") == "x = 1\n"

    def test_stray_closing_fence_is_dropped(self):
        """Observed from this model in the framework benchmark."""
        assert llm.strip_code_fences("x = 1\n```") == "x = 1\n"

    def test_longest_block_wins(self):
        text = "```\nshort\n```\nprose\n```python\nlonger = 1\nmore = 2\n```"
        assert llm.strip_code_fences(text) == "longer = 1\nmore = 2\n"

    def test_json_is_extracted_from_prose(self):
        assert llm.extract_json('sure: {"a": 1} done') == '{"a": 1}'

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert llm.extract_json('{"a": "}{"} tail') == '{"a": "}{"}'

    def test_no_object_returns_none(self):
        assert llm.extract_json("no json here") is None


class TestPytestIsolation:
    def test_workspace_ignores_an_enclosing_pytest_config(self, tmp_path):
        """The agency runs inside a repo that has its own pytest config. If the
        subprocess adopts it, the generated suite silently collects nothing --
        which is exactly what happened before `-c` was pinned."""
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['nowhere']\nnorecursedirs = ['ws']\n"
        )
        root = tmp_path / "ws"
        files = {
            "mod.py": "def double(x):\n    return x * 2\n",
            "tests/test_mod.py": "from mod import double\n\ndef test_double():\n    assert double(2) == 4\n",
        }
        ws = Workspace(root)
        for path, content in files.items():
            ws.write(path, content)
        report = tester.tester_node({"workspace": str(root), "files": files})["test_report"]
        assert report["ok"] is True, report["output"]
        assert report["passed"] == 1

    def test_collection_failure_is_its_own_stage(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.write("mod.py", "x = 1\n")
        report = tester.tester_node(
            {"workspace": str(tmp_path), "files": {"mod.py": "x = 1\n"}, "plan": {"test_file": "tests/t.py"}}
        )["test_report"]
        assert report["stage"] == "collection"


def test_collection_failure_routes_to_replan():
    out = supervisor_node(
        {
            "test_report": {"ok": False, "stage": "collection", "passed": 0, "failed": 0},
            "review": review(),
            "iteration": 1,
            "replans": 0,
        }
    )
    assert out["decision"] == "replan"
