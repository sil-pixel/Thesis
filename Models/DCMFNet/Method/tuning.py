#!/usr/bin/env python3
"""Compatibility launcher for the historical Optuna workflow."""

from runpy import run_module


if __name__ == "__main__":
    run_module("legacy.tuning", run_name="__main__")
