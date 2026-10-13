import os
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Source(BaseModel):
    source: str = Field(min_length=1)
    explanation: str


PROMPTS = {
    "orchestrator": "Choose the next specialist dynamically from the dialogue. Return next_role, instruction, reason. You only route: write no theory or code. Finish only if verified execution is true. At most 8 decisions. Send mathematical plans to scientist and tracebacks to executor.",
    "scientist": "Own the scientific plan. Give y''+a*y'+b*y+c*y^3=g*u, state y'=v,v'=alpha_theta(y,v,u), a fully connected acceleration network, scaling, RK4 input interpolation, optimizer and training windows, and numbered implementation steps. Evaluate a single 4096-sample free run initialized only at sample zero with measured y0 and finite difference velocity. Train first 3072 and test final 1024 without resets or test y inputs. Do not invent measurements. No coding tool.",
    "general": "Answer only the supporting conceptual question. Do not write the fitting plan, code, or invented measurements. No coding tool.",
    "executor": "Implement the latest scientist plan or repair its last implementation. Return complete Python source and explanation. Your source runs in a dedicated directory, may read task5 modules and data/SNLS80mV.mat, and must write candidate.pt containing {'config': {'widths': [...]}, 'state_dict': Acceleration(...).state_dict()}. Use task5.model.Acceleration or compatible parameter names. Training configuration can be stored too. The trusted evaluator will actually run your checkpoint over all 4096 samples and print test RMSE. Do not copy an existing checkpoint or claim fabricated metrics. No subprocess or network. CPU only, one thread, five-minute execution budget. Dataset path is ROOT/data/SNLS80mV.mat, obtain ROOT from task5.data.__file__. If implementing custom training, use this absolute dataset path. Write files only into your current run directory.",
}

PROMPTS["scientist"] += (
    " The execution/evaluation contract uses task5.model.Acceleration(widths): "
    "fully connected tanh hidden layers, linear scalar output times 20000, "
    "inputs scaled by (0.1,45,0.1). Choose the hidden widths/depth and training "
    "settings within this supported model family. The executor saves candidate.pt."
)
PROMPTS["executor"] += (
    " ROOT is Path(task5.data.__file__).resolve().parents[1], not the module file itself. "
    "Do not call task5.train.train() directly: its dataset/output paths assume the "
    "project root, whereas your working directory is isolated. Use load(ROOT/'data/SNLS80mV.mat') "
    "and implement the scientist's training loop in your source."
)


class ModelAgents:
    def __init__(self):
        self.calls_completed = 0
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            timeout=90,
            max_retries=1,
            max_tokens=6000,
        )

    def call(self, role, state):
        import json

        messages = [("system", PROMPTS[role]), ("human", json.dumps(state))]
        if role == "orchestrator":
            result = self.model.with_structured_output(Decision).invoke(messages)
        elif role == "executor":
            result = self.model.with_structured_output(Source).invoke(messages)
        else:
            result = self.model.invoke(messages).content
        self.calls_completed += 1
        return result
