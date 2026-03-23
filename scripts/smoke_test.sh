#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m mgvs.cli.main
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -q
