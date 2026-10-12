"""Unit tests for Milestone M13: Full Evaluation and Engineering Acceptance Criteria."""

import pytest
import os
import json
from pathlib import Path
from evaluation.benchmark_suite import run_full_benchmark


def test_m13_acceptance_criteria():
    results = run_full_benchmark()
    assert results["robot_stability"]["status"] == "PASSED"
    assert results["computer_vision"]["status"] == "PASSED"
    assert results["slam_accuracy"]["status"] == "PASSED"
    assert results["nlp_pipeline"]["status"] == "PASSED"
    assert results["neural_deployment"]["status"] == "PASSED"
    assert results["end_to_end_demonstration"]["status"] == "PASSED"
    assert results["dataset_hub"]["status"] == "PASSED"

    # Verify generated report files
    repo_root = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D")
    assert (repo_root / "reports" / "engineering_report.json").exists()
    assert (repo_root / "reports" / "ENGINEERING_REPORT.md").exists()
