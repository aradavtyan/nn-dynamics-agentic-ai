# Homework 1 — Ara Davtyan

Solutions and executed experiments for ReLU networks, parameter counting,
softmax, attention permutations, a neural Silverbox acceleration model, and
a four-role LangGraph system. The written work is in `report/homework1.pdf`.

Tasks 1–5 have real numerical checks and Silverbox results. Task 6 infrastructure
is tested. Its latest real API requests were blocked by an exhausted API credit
balance (HTTP 429); no model-generated decision or executor script completed.
GitHub publication is pending an authenticated repository destination.

## Layout

`task1`–`task4`: numerical mathematical checks; `task5`: dataset, model,
RK4, training and evaluation; `task6`: roles, graph, execution tool and CLI;
`tests`: mathematical, dynamics and graph tests; `results`: measured records,
checkpoints, plots and trace; `report`: reproducible PDF generator and report.

## Installation

Tested on Linux with Python 3.13.9, CPU PyTorch. A working GPU is not required.
Run all commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` pins the main dependencies. The report generator uses system
DejaVu fonts, normally installed on Linux.

## Data

```bash
python -m task5.data
```

The official `nonlinear-benchmarks` loader downloads the Silverbox archive linked
from https://www.nonlinearbenchmark.org/benchmarks/silverbox. The raw file is copied
to `data/SNLS80mV.mat`. An existing MAT file is reused without another download.
Its variables, SHA-256, and zero baseline are recorded in
`results/dataset.json`. Alternatively put that exact MAT file at that path.
V1 is u, V2 is y. This homework uses its own `[42650:46746]` slice, not the
loader's default benchmark split. Both channels are centered over all 4096
samples, as required. Only the first 3072 samples enter training.

## Mathematical checks and tests

```bash
python -m task1.verify_networks
python -m task2.parameter_count
python -m task3.verify_softmax
python -m task4.verify_attention
python -m pytest -q
```

Dataset tests require the actual downloaded file. Mock roles in unit tests are
infrastructure checks only; they are not Task 6 model-driven experiments.

## Training and evaluation

```bash
python -m task5.experiments
python -m task5.train --widths 32 32 --steps 600 --horizon 32 --lr 0.001 --name manual
python -m task5.evaluate results/w32_lr0.001_h32.pt
```

The experiment command resumes completed runs only when their checkpoint exists,
their configuration matches, and reevaluation agrees with their JSON metrics.
Missing checkpoints are retrained after cloning; invalid records are preserved
under `results/history`. Failed runs are retried. To repeat
an experiment from scratch, use the training CLI with a new name, or move its
JSON record aside before rerunning the sweep. The selected checkpoint is included
in Git; other generated checkpoints are excluded and can be retrained. Summary JSON and plots remain
available for review. All scores come from an uninterrupted 4096-point rollout
with one initial state. The predictor accepts u and only y[0], y[1], so there is
no test-boundary reset or measured test-output input.

Training uses derivative-estimate warm-up followed by differentiable multistep
RK4 window losses. Input is linearly interpolated between samples. Tuning selects
the lowest **test** RMSE, as the assignment requests; that selected test score is
not an untouched independent generalization estimate. Saved experiment records
specify optimizer, rate, step count, rollout, seed, wall time and measured errors.

## Agent system

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` locally. Configure
`OPENAI_MODEL` and optionally `OPENAI_BASE_URL` for a compatible endpoint.

```bash
python -m task6.run
```

This calls real models and can incur API charges. The orchestrator makes up to
eight structured routing decisions; each specialist reads the chronological
dialogue. Only the executor receives the execution tool. It writes a generated
Python source file, runs it, and returns actual exit status, stdout and stderr.
The script must save a compatible `candidate.pt`; a separately executed trusted
evaluator prints and validates its full-free-run RMSE before finish is permitted.
Each complete tool call, including evaluation, has a 300-second budget.

Execution uses a dedicated directory, stripped credentials, Python audit path
checks, blocked subprocess/network calls, process-group termination, and Linux
CPU/memory/file limits. These are practical safeguards, **not a security sandbox
for hostile code**: native extensions can bypass Python auditing. Bubblewrap is
installed here but user namespaces are disabled. Use a container or VM for
untrusted model execution. All execution artifacts and errors remain under
`results/agent_runs`; the graph trace is `results/task6_trace.json`.

The stored real invocation reached OpenAI but returned `credit_balance_exhausted`.
A configured key also needs available API credits. Resolve that account limitation
and rerun successfully before Task 6 can be called complete.
Unit tests additionally execute real trivial scripts, including errors and a
checkpoint evaluator; none of those runs count as the required agent demonstration.

## Report and submission

```bash
python -m report.build
```

The report reads measured numbers from saved JSON and embeds the selected
checkpoint's plot after reevaluating it and checking its saved metrics.
No repository URL is invented. Publish Tasks 5 and 6 to your GitHub repository,
then supply its verified URL and regenerate/update the report before submission.
The source assignment requests one PDF and a GitHub code link. Email submission
has not been performed. AI assistance was used for implementation, proofs,
experiments, tests and report generation; the student should review the work and
follow any lecturer disclosure rules.

## Technical audit

`report/verification.md` records the final audit and its remaining blockers.
Dependencies were installed and the suite run in a separate clean environment,
now retained as `.venv`. A same-seed training repeat produced identical weights and metrics.
The independent replay reads MAT channels and checkpoint tensors directly and
uses its own RK4 loop without importing `task5`:

```bash
python -m task5.train --widths 32 32 --steps 600 --horizon 32 --lr 0.001 --name audit_repeat_best
python -m audit.verify_results
```

The replay independently evaluates every retained checkpoint against its saved
metrics. Only the selected checkpoint is retained in the minimal submission.
Run the experiment sweep to regenerate the other models; run the repeat command
above to repeat the determinism check. This does not replace the real Task 6 workflow.

## Required result files

The homework requires measured architecture comparisons, the selected model and
its configuration, the final test plot, and Task 6 execution evidence. `results/`
therefore contains only:

- `experiments.json`: measured comparison settings and errors.
- `best.json`: selected architecture, training configuration and final metrics.
- `w32_lr0.001_h32.pt`: selected model weights and normalization buffers.
- `w32_lr0.001_h32.png`: required measured-versus-predicted test plot.
- `dataset.json`: benchmark identity, checksum and baseline.
- `task6_trace.json`: actual Task 6 status and the API credit-balance error.

Other checkpoints, prediction arrays, repeat artifacts and generated test/audit
reports are omitted. All can be regenerated from the source and commands above.
