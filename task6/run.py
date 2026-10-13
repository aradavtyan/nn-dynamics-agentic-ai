import json
import os
from pathlib import Path

from dotenv import load_dotenv

from task6.agents import ModelAgents
from task6.coding_tool import CodingTool
from task6.graph import build

REQUEST = """Solve Task 5 for Silverbox SNLS80mV.mat: V1=input u, V2=output y, fs=610.35 Hz. Use [42650:46746], center each whole excerpt, train first 3072, test last 1024. Model the Duffing acceleration with a fully connected neural network on (y,v,u), integrate y'=v,v'=alpha_theta using RK4. Try multiple architectures and tune optimizer, learning rate, training steps and rollout length. Evaluate one continuous 4096-point free run initialized from y[0] and finite difference velocity; no state resets at index 3072 and no test y as inputs. Report train/test RMSE and zero baseline, compare architectures and plot the test prediction. Have scientist give the plan, executor write and run code. The executable must save candidate.pt for independent trusted free-run measurement."""


def main():
    load_dotenv()
    trace = {"request": REQUEST, "events": [], "status": "running"}
    path = Path("results/task6_trace.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.update(
        graph_compiled=False,
        real_llm_calls_completed=0,
        orchestrator_decisions=0,
        executor_script_created=False,
        executor_script_executed=False,
        exit_code=None,
        test_rmse=None,
        valid_finish_condition_met=False,
    )
    try:
        agents = ModelAgents() if os.getenv("OPENAI_API_KEY") else None
        graph = build(agents, CodingTool()) if agents is not None else None
        if graph is not None:
            trace["graph_compiled"] = True
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is absent; no real model calls can be made"
            )
        initial = {
            "dialogue": [{"role": "user", "text": REQUEST}],
            "decisions": 0,
            "verified": False,
            "executions": [],
        }
        for event in graph.stream(
            initial, {"recursion_limit": 24}, stream_mode="updates"
        ):
            trace["events"].append(event)
            for role, update in event.items():
                trace["real_llm_calls_completed"] = agents.calls_completed
                if "decisions" in update:
                    trace["orchestrator_decisions"] = update["decisions"]
                if role == "executor" and update.get("executions"):
                    result = update["executions"][-1]
                    trace.update(
                        executor_script_created=True,
                        executor_script_executed=True,
                        exit_code=result["exit_code"],
                        test_rmse=result.get("test_rmse"),
                        valid_finish_condition_met=update.get("verified", False),
                    )
            path.write_text(json.dumps(trace, indent=2))
        last = next(iter(trace["events"][-1].values()))
        trace["status"] = last.get("status", "unsuccessful")
    except Exception as exc:
        error = repr(exc)
        if os.getenv("OPENAI_API_KEY"):
            error = error.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
        error_code = getattr(exc, "code", None)
        blocked = not os.getenv("OPENAI_API_KEY") or error_code in {"insufficient_quota", "credit_balance_exhausted"}
        trace.update(
            status="blocked" if blocked else "failed",
            error=error,
            error_code=error_code,
        )
    path.write_text(json.dumps(trace, indent=2))
    print(
        json.dumps(
            {
                "status": trace["status"],
                "trace": str(path),
                "error": trace.get("error"),
            },
            indent=2,
        )
    )
    return 0 if trace["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
