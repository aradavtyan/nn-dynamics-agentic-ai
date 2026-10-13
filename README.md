# Homework 1 — Silverbox oscillator

Ara Davtyan · [Repository](https://github.com/aradavtyan/nn-dynamics-agentic-ai)

This repository contains Task 5 code, tests and measured results. The written
solutions to Tasks 1–4 and the Task 5 explanation and plot are in the separate
homework PDF submitted to the lecturer.

## Files

- `task5/`: data loading, neural acceleration model, RK4, training and evaluation.
- `tests/`: dataset, integration, gradient, metric and free-run evaluation checks.
- `results/`: architecture comparisons, selected model/configuration and test plot.

## Setup

Tested on Linux with Python 3.13.9 and CPU PyTorch. Run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m task5.data
python -m pytest -q
```

The selected model is included in Git. The data are downloaded separately from
[the official Silverbox benchmark](https://www.nonlinearbenchmark.org/benchmarks/silverbox).
An existing `data/SNLS80mV.mat` is reused. No API key is needed.

## Model and evaluation

The model learns acceleration in `y'=v, v'=alpha_theta(y,v,u)` using a fully
connected tanh network and differentiable RK4. Input is linearly interpolated
between samples. V1 is the input voltage, V2 the measured output voltage, and the
sampling rate is 610.35 Hz.

The slice `[42650:46746]` contains 4096 samples. Each channel is centered over
that whole excerpt, then split into 3072 training and 1024 test samples.
Evaluation makes one continuous free run, initialized from the first measured
position and a finite-difference velocity. It does not reset the state at the
test boundary or use measured test outputs as prediction inputs.

```bash
python -m task5.evaluate results/w32_lr0.001_h32.pt
python -m task5.experiments
python -m task5.train --widths 32 32 --steps 600 --horizon 32 --lr 0.001 --name manual
python -m task5.verify_results
```

The selected architecture is 3 → 32 → 32 → 1. Its measured free-run RMSE is
0.002332454 V on training and 0.008616101 V on testing. The zero-prediction test
baseline is 0.060491854 V. Exact metrics, settings and six measured comparison
runs are in `results/experiments.json` and `results/best.json`.
The test plot is `results/w32_lr0.001_h32.png`.

The experiment sweep recreates missing models and validates cached settings and
metrics before reuse. Only the selected checkpoint is retained in Git. The
independent verifier reads MAT channels and saved weight tensors directly and
replays RK4 without using the model's rollout function. Test RMSE was used for
architecture selection as requested; it is not an untouched independent estimate.

## Written submission

For Task 5, the lecturer asks for the retained architecture and hyperparameters,
alternative settings with their measured test errors, both continuous free-run
RMSE values, the test plot and the GitHub repository link in the PDF. The PDF is
submitted separately by email to the address in the assignment; it is not part
of this code repository. This package covers Tasks 1–5.

AI assistance was used for implementation, experiments, testing and report
preparation. The student should review the work and follow applicable disclosure
rules.
