# Final technical verification — Homework 1

Student: Ara Davtyan. Audit date: 13 October 2026.

This is the historical verification report. On the student's request, temporary
audit outputs, comparison checkpoints, prediction arrays and repeated model were
subsequently removed. Statements about those audit runs describe the actual
completed verification, not files still included in the submission. The current
minimal result file list is in README.md. The selected model remains independently
verifiable; other models and repeat artifacts can be regenerated with the documented
training commands. Automated tests remain in the project.

## Overall status

**PARTIAL.** Tasks 1–5 pass mathematical review and real execution checks.
Task 6 infrastructure passes unit and integration tests, including real Python
subprocesses. Real API requests were attempted with the saved local key and a
replacement configuration. Both reached OpenAI and were rejected with HTTP 429,
credit_balance_exhausted. No model-generated decisions completed. GitHub publication and a verified repository URL remain unavailable.
No successful model-driven Task 6 execution is claimed.

## Task-by-task verification

| Task | Status | Executed tests | Issues found | Issues fixed |
|------|--------|----------------|--------------|--------------|
| 1 | PASS | Required absolute-value examples, all Boolean corners, floating-point extremes and int64 minimum; report proof reviewed | Integer minimum could overflow during negation | Convert the absolute-value network input to floating point before negation |
| 2 | PASS | Executed PyTorch convolutions, pools, ReLUs and dense layers; every shape and per-layer count; exact total | No numerical discrepancy | Added independently specified per-layer count regression |
| 3 | PASS | Ordinary, large, tiny and maximally separated finite logits; invalid-input tests | Extreme finite subtraction emitted an overflow warning; invalid logits were accepted | Handle expected subtraction/exp underflow correctly; reject empty or nonfinite logits; retain exact-arithmetic positivity distinction |
| 4 | PASS | Random matrices/permutations across four sizes and key widths; score and softmax identities; required two-token and positional examples | No proof or numerical discrepancy | Added broader independent numerical coverage |
| 5 | PASS | Actual dataset and archive checksum comparison; known-solution RK4 accuracy/order; gradients and optimizer updates; poisoned-test-output regression; six checkpoint replays; full same-seed retraining | Cache trusted JSON without a checkpoint; RMSE silently broadcast unequal shapes; warm-up lacked finite checks; report architecture was hard-coded | Validate cached artifacts/configurations/metrics, retry invalid/failed runs, preserve history, reject invalid RMSE arrays, validate training settings, check warm-up gradients, derive report architecture/count from selected configuration |
| 6 | PARTIAL | Four-node compilation/routing, dialogue history, schema, eight-decision limit, premature/forged finish, model errors, actual successful/failed Python processes, timeout, credential stripping, path/network restrictions, invalid-checkpoint rejection and a successful mocked-role graph with real evaluation | Finish trusted a Boolean without also checking execution evidence; unsupported scientist plan could conflict with checkpoint interface; trace lacked explicit outcome counters | Require matching successful evaluator output and executor evidence; explain the supported model contract to scientist/executor; persist real workflow outcome fields and count only completed model calls |

The mathematical review checked every weight/bias and activation, the two-hidden-unit
minimality arguments without skip connections, the finite-class softmax proof and
continuous-density distinction, and the permutation convention used in the PDF.
No changes to the assigned data split, free-run metric or selection criterion were made.

## Task 5 measured results

| Item | Verified value |
|------|----------------|
| Final architecture | 3 → 32 → 32 → 1, tanh hidden layers, linear acceleration output |
| Trainable parameters | 1,217 |
| Input scales | y / 0.1 V, v / 45 V/s, u / 0.1 V |
| Acceleration scale | 20,000 V/s² |
| Optimizer | Adam, learning rate 0.001 |
| Training | 1,200 derivative warm-up updates + 600 multistep updates; 32-sample horizons; batch size 32; seed 42 |
| Training RMSE | **0.002332453578115664 V** |
| Test RMSE | **0.008616101247742852 V** |
| Zero baseline | **0.060491854406981314 V** |
| Completed architectures | **2**: [16,16] and [32,32] |
| Completed comparison settings | **6**, plus one independent same-seed repeat of the kept setting |

Evaluation command:

```bash
.venv/bin/python -m task5.evaluate results/w32_lr0.001_h32.pt
```

Dataset SHA-256:
`5c7413a52255af0cb4e2f93fb6903beb92b08965ba1e6ad0a6ffedfed8477ed4`.
Before cleanup, the MAT file exactly matched the corresponding member of the downloaded
official archive. It still matches the original saved hash; the redundant archive
and extracted duplicate channels were removed. No publisher-provided checksum was
available, so this verifies local artifact consistency rather than independent
publisher authentication. V1 and V2 each have 131,072 samples. The raw slice
[42650:46746] was separately centered for u and y, producing 4,096 samples, split
3,072/1,024, with dt = 1/610.35 seconds.

`audit/verify_results.py` imports no Task 5 code. It reads MAT variables and saved
network tensors directly, applies the affine/tanh network operations, and performs
its own four-stage RK4 loop. It starts once from y[0] and (y[1]-y[0])/dt and uses
only u thereafter. All six 4,096-point predictions matched their saved NPZ arrays
exactly, and every recalculated RMSE exactly matched its JSON record. The plotted
inputs, measured outputs and predictions were checked against those arrays.

The repeat used the clean environment and current audited training code:

```bash
.venv/bin/python -m task5.train --widths 32 32 --steps 600 --horizon 32 --lr 0.001 --name audit_repeat_best
.venv/bin/python -m audit.verify_results
```

Every repeated weight tensor and all three RMSE values were bit-identical to the
original best checkpoint. Original metrics were retained: there was no corrected
score to substitute. Actual optimizer updates and finite gradients were also
verified independently in tests. Selecting the best setting by test RMSE follows
the assignment; that score is not an untouched independent generalization estimate.

## Task 6 execution results

| Item | Real Task 5 workflow | Separate infrastructure evidence |
|------|---------------------|----------------------------------|
| Graph compiled | Yes, before the real API request | Yes; exactly four role nodes, excluding START/END |
| Real LLM calls completed | **0** | Mock roles only; not LLM evidence |
| Executor script created | No | Yes, real new Python files |
| Executor script executed | No | Yes, actual subprocesses |
| Exit code | CLI **1** for exhausted API credits; no executor exit code | Successful processes returned 0; failures returned nonzero status with actual errors |
| Test RMSE printed | No | Trusted evaluator printed a real metric for an untrained test checkpoint; not a fitted agent result |
| Valid finish condition met | **No** | Yes in a mocked-role integration test only after actual successful evaluation |
| Orchestrator decisions | **0** | Limit of eight model decisions exercised using mocks |

Actual invocation: `.venv/bin/python -m task6.run`.
Its trace is `results/task6_trace.json`; redundant console-output copies were removed.
No role transition or generated scientific
plan exists for the real run, and none was invented. Schema/model errors fail
without producing success. A textual RMSE or forged verified flag cannot finish.
Repair turns preserve previous source files and actual error traces. The trusted
evaluator fixes the required acceleration model and continuous free-run protocol;
compliance of a real LLM-generated training plan remains unvalidated.

The subprocess timeout mechanism was exercised with a shortened test timeout;
the production constant is 300 seconds. CPU, file, memory and file-descriptor
limits plus Python audit path/network checks are present. These checks are not a
security boundary against hostile native extensions. No stronger isolation is
claimed; user namespaces were unavailable in the earlier environment check.

## Tests

- Total automated tests: **60**.
- Passed: **60**.
- Failed: **0**.
- Skipped: **0**.
- Blocked external validation: **1** real model-driven Task 6 workflow, outside
  the 60-test suite. GitHub publication is a separate blocked deliverable.

The original 20-test suite was rerun before corrections and passed. After targeted
corrections, expanded coverage and integration tests passed in a separately
created clean environment installed from `requirements.txt` with the
documented CPU index. `pip check` reported no broken requirements. Every source
module imported successfully. Ruff checks passed. The final suite output is
`results/pytest.txt`, with machine-readable counts in `results/audit/environment.json`.

Executed entry points included dataset acquisition (using the official cached
download), all four mathematical checks, the experiment sweep with validated
cache reuse, fresh training, checkpoint evaluation, the independent audit,
Task 6 preflight and report generation. The verified environment was consolidated
into `.venv`; duplicate installation logs and intermediate test reports were removed.
Essential summaries remain under `results/audit`. The environment uses Python 3.13.9
and CPU PyTorch 2.14.1+cpu; no working CUDA backend was available.

## Corrections

1. Cached experiment records now require actual compatible checkpoints and fresh
   metric agreement. Missing artifacts trigger training; failed/invalid records
   are retried and preserved in history. An all-failed sweep cannot retain a
   stale best-result file.
2. RMSE rejects empty, mismatched and nonfinite arrays rather than broadcasting
   targets. Softmax handles extreme finite subtraction without misleading
   warnings and rejects invalid scores. Absolute value avoids signed-integer
   minimum overflow.
3. Training validates horizons, update counts, names, rates and batch sizes;
   warm-up checks finite losses/gradients without altering valid updates. Output
   directories are created as needed; optimizer selection is exposed in the CLI.
4. Agent success requires successful process/evaluator evidence and a matching
   finite printed metric, not only a Boolean. Decision fields reject whitespace
   and extra fields. The scientist/executor prompts explain the checkpoint/model
   contract and isolated working-directory paths. Real trace outcome counters
   distinguish preflight failures from actual executions.
5. The report derives its selected architecture and parameter count dynamically,
   reevaluates the checkpoint before using its saved metrics, refreshes its plot,
   accurately describes warm-up versus rollout clipping, and records audit
   evidence. Matplotlib uses a writable temporary cache. Audit environment and
   submission archives are excluded from Git.
6. Added independent and regression coverage described above. No failing test was
   deleted and no experimental value was fabricated or improved by changing the
   evaluation protocol.

## Remaining limitations

- A key is configured locally, but OpenAI returned credit_balance_exhausted on
  the real requests. Real model-generated plans and end-to-end Task 6 performance
  remain unverified. A key shared in chat should be revoked and replaced locally.
- No usable Git metadata, remote, authenticated GitHub account or destination
  was available. No push succeeded or was claimed; the PDF correctly identifies
  the missing repository link. Publication remains required for submission.
- The first four original checkpoints predate provenance metadata. Their source
  version cannot be recovered from the checkpoints alone. Their configurations,
  artifacts and errors were independently checked, and the best run was fully
  retrained with current source/provenance metadata and identical results.
- CPU-only execution was tested. CUDA portability and performance were not tested.
- No email submission was performed.

## Final deliverables

Paths are relative to `/home/ara/Desktop/AI-Course-homeworks/homework_1`:

- Source code: `task1/` through `task6/`; independent checker: `audit/`.
- Saved selected model: `results/w32_lr0.001_h32.pt`.
- Repeated model: `results/audit_repeat_best.pt`.
- Experiment results: `results/experiments.json` and `results/best.json`.
- Independent verification: `results/audit/independent_results.json`.
- Test plot: `results/w32_lr0.001_h32.png`.
- Tests: `results/pytest.txt` and `results/audit/environment.json`.
- Real Task 6 blocker trace: `results/task6_trace.json`.
- Final PDF: `report/homework1.pdf`.
- This verification report: `report/verification.md`.
- Submission archive: omitted from the GitHub project; the source and final PDF are retained.
- GitHub repository: **not published; no verified URL**.

All eight PDF pages were rendered and visually inspected after regeneration.
Equations, tables, voltage/time labels, plot, footers and boundary margins were
checked. The PDF records the current test outcome and measured metrics and has
no invented Task 6 trace or repository link. Missing external requirements remain
clearly identified, so the overall submission is not declared complete.

## Cleanup

The final package retains all task source, tests, dependency files, the raw MAT
record, six comparison checkpoints/configurations/prediction arrays, one final
plot, the repeated checkpoint/configuration, essential audit summaries, the real
Task 6 blocker trace and the final PDF. All retained numerical artifacts support
execution or independent verification. Redundant download copies, other benchmark
channels, non-selected/repeated plots, intermediate logs/test reports, obsolete
acceptance notes and Python/tool caches were removed. The two virtual environments
were consolidated into the verified CPU environment at `.venv`, excluded from
the submission archive and Git. Protected workspace metadata and secrets were
left untouched. No valid Git repository was present, so tracked-file checks could
not identify any tracked project files; no Git history or remote was changed.

Latest Task 6 testing also verified that quota failures are recorded as external
blockers and API error logging redacts the configured key. The full current suite
passed 60 tests. The optional requirements lock file was removed; main dependencies
remain pinned in requirements.txt. Empty run directories and transient test output
were omitted from the final submission. No secrets were copied into the archive.
