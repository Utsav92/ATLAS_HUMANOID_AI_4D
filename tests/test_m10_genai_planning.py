"""Unit tests for Milestone M10: Generative AI Planning & Explanation."""

import pytest
from atlas.genai.reasoning import LocalGenerativePlanner
from atlas.core.types import EpisodeLog


def test_m10_rag_retrieval():
    planner = LocalGenerativePlanner()
    docs = planner.search_docs("How to recover balance when falling?", top_k=1)
    assert len(docs) == 1
    assert "Balance Recovery" in docs[0]["title"]


def test_m10_schema_validation():
    planner = LocalGenerativePlanner()
    valid_json = {
        "task": "PICK_AND_PLACE",
        "object": "red_cube",
        "destination": "shelf",
        "steps": ["LOCATE", "GRASP", "PLACE"]
    }
    ok, _ = planner.validate_plan_schema(valid_json)
    assert ok is True

    bad_json = {"task": "INVALID_TASK"}
    bad_ok, err = planner.validate_plan_schema(bad_json)
    assert bad_ok is False


def test_m10_hallucination_detection():
    planner = LocalGenerativePlanner()
    known = ["table", "shelf", "red_cube"]

    plan_valid = {"object": "red_cube"}
    is_hall, _ = planner.detect_hallucinations(plan_valid, known)
    assert is_hall is False

    plan_hall = {"object": "floating_gold_coin"}
    is_hall2, msg = planner.detect_hallucinations(plan_hall, known)
    assert is_hall2 is True
    assert "Hallucinated object" in msg


def test_m10_generate_and_explain():
    planner = LocalGenerativePlanner()
    cmd = "Walk to the red cube, pick it up, and place it on the shelf."
    plan = planner.generate_task_plan(cmd, known_world_objects=["red_cube", "table", "shelf"])

    assert plan.task == "PICK_AND_PLACE"
    assert plan.status == "PENDING"
    assert "rag_doc_id" in plan.metadata

    log = EpisodeLog(episode_id="ep_test_01", instruction=cmd, success=True)
    log.explanations = ["Step 1", "Step 2"]
    log.timestamps = [0.0, 0.1]
    explanation = planner.explain_episode(log)
    assert "SUCCESS" in explanation
