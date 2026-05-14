"""Entry point for PODE experiments.

Usage:
    python solve.py

This is a thin wrapper that delegates to experiments.py.
The code has been split into focused modules:
    - problems.py:    ODE/PDE problem definitions (BaseODE, linear, exp_test, etc.)
    - model.py:       DenseNet neural network
    - curriculum.py:  CurriculumClass (curriculum learning schedule)
    - trainer.py:     Trainer class (training loop, loss functions, plotting, checkpointing)
    - experiments.py: Experiment configs and main() entry point
"""
from experiments import main

if __name__ == "__main__":
    main()
