#!/usr/bin/env python3
"""Compatibility launcher for :mod:`dcmfnet.api`."""

from dcmfnet.api import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
