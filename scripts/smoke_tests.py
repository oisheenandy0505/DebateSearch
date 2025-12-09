#!/usr/bin/env python3
"""Quick smoke tests for DebateSearch: quality_gate + health endpoint."""

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.utils.quality import quality_gate  # noqa: E402

def test_quality_gate():
    cues = {"favor": {"freedom": 1.0}, "against": {}}
    stop = {"the", "and", "of"}
    score = quality_gate("Freedom for all citizens.", cues, stop)
    assert score[0] > 0.5 and score[2], "quality_gate should flag this as stanceable"

def test_health():
    resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
    data = resp.json()
    assert resp.ok and data.get("status") in {"ok", "degraded"}

if __name__ == "__main__":
    test_quality_gate()
    test_health()
    print("Smoke tests passed.")
