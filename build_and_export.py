#!/usr/bin/env python3
"""Run local training and export the inference artifact."""
from __future__ import annotations
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "train.py", *sys.argv[1:]], check=True)
    subprocess.run([sys.executable, "export_model.py"], check=True)
