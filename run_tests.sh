#!/usr/bin/env bash
# Phase 3 verified test suite. T1-T11 must pass before any Phase 4 experiment.
#
#   ./run_tests.sh            full suite, one line per test
#   ./run_tests.sh blocking   only the blocking (starred) tests
#   ./run_tests.sh T7         one group
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-all}" in
  blocking) exec python3 -m pytest tests/ -v -k \
      "t7_5 or t2_3 or t7_2e or t3_1 or t3_2 or t3_3 or t4_1 or t4_2 or t6_3" ;;
  all)      exec python3 -m pytest tests/ -v ;;
  *)        exec python3 -m pytest tests/ -v -k "${1,,}" ;;
esac
