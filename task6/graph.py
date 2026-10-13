import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from task6.agents import Decision, Source
from task6.coding_tool import verified_execution


class State(TypedDict, total=False):
    dialogue: list[dict]
    decisions: int
    next_role: str
    verified: bool
    status: str
    executions: list[dict]


def build(agents, tool):
    def orchestrator(state):
        if state.get("decisions", 0) >= 8:
            return {"next_role": "finish", "status": "unsuccessful: decision limit"}
        decision = Decision.model_validate(agents.call("orchestrator", state))
        count = state.get("decisions", 0) + 1
        dialogue = state["dialogue"] + [
            {"role": "orchestrator", "text": json.dumps(decision.model_dump())}
        ]
        status = "running"
        role = decision.next_role
        if role == "finish":
            evidence = state.get("executions", [])
            status = (
                "success"
                if state.get("verified", False)
                and evidence
                and verified_execution(evidence[-1])
                and any(m["role"] == "executor" for m in state["dialogue"])
                else "unsuccessful: unverified finish rejected"
            )
        return dict(dialogue=dialogue, decisions=count, next_role=role, status=status)

    def specialist(role):
        def node(state):
            response = agents.call(role, state)
            return {
                "dialogue": state["dialogue"] + [{"role": role, "text": str(response)}]
            }

        return node

    def executor(state):
        if not any(m["role"] == "scientist" for m in state["dialogue"]):
            return {
                "dialogue": state["dialogue"]
                + [
                    {
                        "role": "executor",
                        "text": "Execution requires a scientist plan first.",
                    }
                ]
            }
        response = Source.model_validate(agents.call("executor", state))
        result = tool.execute(response.source)
        return {
            "dialogue": state["dialogue"]
            + [
                {
                    "role": "executor",
                    "text": json.dumps(
                        {
                            "source": response.source,
                            "explanation": response.explanation,
                            "execution": result,
                        }
                    ),
                }
            ],
            "executions": state.get("executions", []) + [result],
            "verified": verified_execution(result),
        }

    graph = StateGraph(State)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("scientist", specialist("scientist"))
    graph.add_node("general", specialist("general"))
    graph.add_node("executor", executor)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda s: s["next_role"],
        {r: r for r in ["scientist", "general", "executor"]} | {"finish": END},
    )
    for role in ["scientist", "general", "executor"]:
        graph.add_edge(role, "orchestrator")
    return graph.compile()
