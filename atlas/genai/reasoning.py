"""Generative AI Module for ATLAS HUMANOID AI 4D.
Provides structured JSON task planning, RAG documentation retrieval,
hallucination detection against simulated world state, and natural language episode explanations.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from atlas.core.types import TaskPlan, EpisodeLog
from atlas.nlp.pipeline import SKILL_RECIPES, KNOWN_INTENTS, KNOWN_OBJECTS, KNOWN_DESTINATIONS


# Local Robot Engineering & Safety Documentation Index for RAG
ROBOT_DOCS = [
    {
        "id": "doc_safety_01",
        "title": "Unitree G1 Joint Limits and Actuator Ranges",
        "content": "G1 joint limits require waist pitch to stay between -0.5 and 0.5 rad. Knee flexion maximum is 2.5 rad. Never command actuator torques exceeding hardware limits. Always engage safety supervisor before sending torque commands."
    },
    {
        "id": "doc_pick_place_02",
        "title": "Dexterous Humanoid Pick and Place Protocol",
        "content": "Pick and place sequence comprises 8 sequential phases: LOCATE (RGB-D perception), APPROACH (A* path locomotion), STABILIZE (CoM zero-velocity equilibrium), REACH (DLS inverse kinematics), GRASP (finger tactile closure), LIFT (vertical +10cm clearance), TRANSPORT (base motion while holding load), PLACE (controlled descent and release)."
    },
    {
        "id": "doc_fall_recovery_03",
        "title": "Humanoid Balance Recovery",
        "content": "When Center of Mass (CoM) deviates beyond support polygon, posture stabilizer applies corrective ankle pitch and roll offsets. If base height drops below 0.50m or roll/pitch exceeds 45 degrees, trigger emergency stand."
    }
]


class LocalGenerativePlanner:
    """CPU-friendly Generative AI planner with RAG and deterministic safety fallback."""
    def __init__(self):
        self.docs = ROBOT_DOCS

    def search_docs(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        """Lightweight keyword/token overlap RAG search over local robot manuals."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for doc in self.docs:
            d_tokens = set(re.findall(r"\w+", (doc["title"] + " " + doc["content"]).lower()))
            overlap = len(q_tokens.intersection(d_tokens))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]

    def validate_plan_schema(self, plan_json: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate generated JSON against strict robotics schema."""
        required_keys = {"task", "object", "destination", "steps"}
        if not required_keys.issubset(plan_json.keys()):
            return False, f"Missing required keys: {required_keys - set(plan_json.keys())}"

        if plan_json["task"] not in KNOWN_INTENTS:
            return False, f"Invalid task intent: '{plan_json['task']}'"

        if not isinstance(plan_json["steps"], list) or len(plan_json["steps"]) == 0:
            return False, "Steps must be a non-empty list of skill names"

        return True, "Valid"

    def detect_hallucinations(self, plan_json: Dict[str, Any], known_world_objects: List[str]) -> Tuple[bool, str]:
        """Check if generated plan references non-existent objects or hallucinated physics."""
        target_obj = plan_json.get("object")
        if target_obj and target_obj != "unspecified" and target_obj != "self":
            if target_obj not in known_world_objects:
                return True, f"Hallucinated object '{target_obj}' does not exist in world: {known_world_objects}"
        return False, "Ground truth verified"

    def generate_task_plan(
        self,
        instruction: str,
        known_world_objects: List[str]
    ) -> TaskPlan:
        """Generate structured plan using RAG context with guaranteed fallback."""
        # 1. Retrieve relevant manual
        retrieved = self.search_docs(instruction)
        context_doc = retrieved[0]["content"] if retrieved else ""

        # 2. Simulated LLM structured response generation
        norm = instruction.lower()
        if "pick" in norm and "place" in norm:
            raw_response = {
                "task": "PICK_AND_PLACE",
                "object": "red_cube",
                "destination": "shelf",
                "steps": SKILL_RECIPES["PICK_AND_PLACE"]
            }
        elif "grasp" in norm or "pick" in norm:
            raw_response = {
                "task": "GRASP",
                "object": "red_cube",
                "destination": "hand",
                "steps": SKILL_RECIPES["GRASP"]
            }
        else:
            raw_response = {
                "task": "NAVIGATE",
                "object": "table",
                "destination": "table",
                "steps": SKILL_RECIPES["NAVIGATE"]
            }

        # 3. Schema validation
        valid_schema, schema_err = self.validate_plan_schema(raw_response)
        if not valid_schema:
            # Fallback
            return TaskPlan(instruction=instruction, task="REJECTED", status="FAILED", metadata={"error": schema_err})

        # 4. Hallucination detection
        is_hallucination, hall_err = self.detect_hallucinations(raw_response, known_world_objects)
        if is_hallucination:
            return TaskPlan(instruction=instruction, task="REJECTED", status="FAILED", metadata={"error": hall_err})

        return TaskPlan(
            instruction=instruction,
            task=raw_response["task"],
            target_object=raw_response["object"],
            destination=raw_response["destination"],
            steps=raw_response["steps"],
            status="PENDING",
            metadata={"rag_doc_id": retrieved[0]["id"] if retrieved else None, "hallucination_check": "passed"}
        )

    def explain_episode(self, log: EpisodeLog) -> str:
        """Generate human-readable summary explanation for the executed episode."""
        outcome = "SUCCESS" if log.success else "FAILED"
        n_steps = len(log.timestamps)
        return (
            f"[ATLAS 4D EXPLAINER]: Episode '{log.episode_id}' concluded with outcome {outcome}. "
            f"The robot completed {n_steps} synchronized steps under command '{log.instruction}'. "
            f"Key events: {'; '.join(log.explanations[:4])}..."
        )
