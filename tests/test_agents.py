import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from task6.agents import Decision
from task6.coding_tool import TIMEOUT, CodingTool
from task6.graph import build


class FakeAgents:


    def __init__(self, roles):
        self.roles = iter(roles)
        self.seen = []

    def call(self, role, state):
        self.seen.append((role, state["dialogue"].copy()))
        if role == "orchestrator":
            return dict(
                next_role=next(self.roles),
                instruction="Do the assigned work",
                reason="Infrastructure test",
            )
        if role == "executor":
            return dict(
                source="print('TEST_RMSE=0.001')",
                explanation="Mock role, actual trivial script",
            )
        return "Mock specialist plan"


def initial():
    return dict(
        dialogue=[{"role": "user", "text": "Test"}],
        decisions=0,
        verified=False,
        executions=[],
    )


def test_schema():
    with pytest.raises(ValidationError):
        Decision(next_role="other", instruction="x", reason="y")
    with pytest.raises(ValidationError):
        Decision(next_role="finish", instruction="", reason="y")


def test_nodes_routes_shared_dialogue_and_no_false_success(tmp_path):
    agents = FakeAgents(["scientist", "general", "executor", "finish"])
    graph = build(agents, CodingTool(tmp_path))
    assert {"orchestrator", "scientist", "general", "executor"} <= set(
        graph.get_graph().nodes
    )
    state = graph.invoke(initial())
    assert [m["role"] for m in state["dialogue"]] == [
        "user",
        "orchestrator",
        "scientist",
        "orchestrator",
        "general",
        "orchestrator",
        "executor",
        "orchestrator",
    ]
    assert state["status"].startswith("unsuccessful") and not state["verified"]
    assert state["executions"][0]["stdout"].strip() == "TEST_RMSE=0.001"
    assert any(m["role"] == "scientist" for m in agents.seen[3][1])
    assert not hasattr(agents, "tool")


def test_eight_decisions(tmp_path):
    agents = FakeAgents(["general"] * 8)
    state = build(agents, CodingTool(tmp_path)).invoke(
        initial(), {"recursion_limit": 24}
    )
    assert state["decisions"] == 8
    assert sum(role == "orchestrator" for role, _ in agents.seen) == 8
    assert state["status"] == "unsuccessful: decision limit"


def test_finish_requires_execution(tmp_path):
    state = build(FakeAgents(["finish"]), CodingTool(tmp_path)).invoke(initial())
    assert state["status"] == "unsuccessful: unverified finish rejected"


def test_actual_write_execute_capture_error(tmp_path):
    tool = CodingTool(tmp_path)
    result = tool.execute(
        "import sys\nprint('hello')\nprint('error stream',file=sys.stderr)\nraise ValueError('real failure')"
    )
    assert (Path(result["directory"]) / "solution.py").exists()
    assert result["exit_code"] != 0 and result["stdout"] == "hello\n"
    assert (
        "error stream" in result["stderr"]
        and "ValueError: real failure" in result["stderr"]
    )
    assert not result["verified"]


def test_timeout(tmp_path):
    assert TIMEOUT == 300
    with pytest.raises(ValueError):
        CodingTool(tmp_path, 301)
    result = CodingTool(tmp_path, timeout=0.1).execute("while True: pass")
    assert result["timed_out"] and result["exit_code"] != 0


def test_path_restriction(tmp_path):
    result = CodingTool(tmp_path).execute(
        "open('/tmp/agent_outside.txt','w').write('forbidden')"
    )
    assert result["exit_code"] != 0 and "PermissionError" in result["stderr"]


def test_no_plan_no_execution(tmp_path):
    state = build(FakeAgents(["executor", "finish"]), CodingTool(tmp_path)).invoke(
        initial()
    )
    assert not state["executions"] and not state["verified"]


def test_checkpoint_independently_evaluated(tmp_path):
    source = """import torch
from task5.model import Acceleration
torch.set_num_threads(1)
model=Acceleration([4])
for p in model.parameters(): p.data.zero_()
torch.save({'config':{'widths':[4]},'state_dict':model.state_dict()},'candidate.pt')
"""
    result = CodingTool(tmp_path).execute(source)
    assert result["exit_code"] == 0, result["stderr"]
    assert result["verified"], result
    assert "test_rmse" in json.loads(result["evaluation"]["stdout"])


def test_repair_preserves_previous_source_and_trace(tmp_path):
    tool = CodingTool(tmp_path)
    first = tool.execute("raise RuntimeError('first failure')")
    second = tool.execute("print('repair succeeded')")
    assert first["directory"] != second["directory"]
    assert "first failure" in (Path(first["directory"]) / "solution.py").read_text()
    assert first["exit_code"] != 0 and second["exit_code"] == 0
    assert (Path(first["directory"]) / "execution.json").exists()


def test_quota_error_is_recorded_as_external_blocker(monkeypatch, tmp_path):
    from task6 import run

    class QuotaError(RuntimeError):
        code = "credit_balance_exhausted"

    def unavailable():
        raise QuotaError("No API credits remaining")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-dummy")
    monkeypatch.setattr(run, "load_dotenv", lambda: None)
    monkeypatch.setattr(run, "ModelAgents", unavailable)
    assert run.main() == 1
    trace = json.loads(Path("results/task6_trace.json").read_text())
    assert trace["status"] == "blocked"
    assert trace["error_code"] == "credit_balance_exhausted"
    assert trace["real_llm_calls_completed"] == 0
    assert not trace["valid_finish_condition_met"]


def test_api_error_does_not_persist_key(monkeypatch, tmp_path, capsys):
    from task6 import run

    dummy = "test-only-key-not-a-real-credential"

    def unavailable():
        raise RuntimeError("Request failed for " + dummy)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", dummy)
    monkeypatch.setattr(run, "load_dotenv", lambda: None)
    monkeypatch.setattr(run, "ModelAgents", unavailable)
    assert run.main() == 1
    saved = Path("results/task6_trace.json").read_text()
    assert dummy not in saved and "[REDACTED]" in saved
    assert dummy not in capsys.readouterr().out
