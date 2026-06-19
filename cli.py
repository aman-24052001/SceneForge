#!/usr/bin/env python3
"""
Thin CLI entrypoint so `python3 cli.py run --images ... --out ...` works
without needing `python3 -m sceneforge.orchestrator` package gymnastics.
"""
from sceneforge.orchestrator import app

if __name__ == "__main__":
    app()
