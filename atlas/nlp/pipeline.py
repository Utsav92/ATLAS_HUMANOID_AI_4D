"""Natural Language Processing and Instruction Parsing Pipeline for ATLAS HUMANOID AI 4D.
Extracts intents, entities, validates plans against robot capabilities, and generates executable TaskPlans.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
from atlas.core.types import TaskPlan


# Allowed canonical vocabularies
KNOWN_INTENTS = {"PICK_AND_PLACE", "GRASP", "NAVIGATE", "STAND", "INSPECT", "WAVE"}
KNOWN_OBJECTS = {"red_cube", "blue_cube", "cube", "table", "shelf", "box", "ball", "cylinder"}
KNOWN_DESTINATIONS = {"shelf", "table", "ground", "bin", "user", "origin"}

# Skill steps decomposed for each intent
SKILL_RECIPES = {
    "PICK_AND_PLACE": [
        "LOCATE",
        "APPROACH",
        "STABILIZE",
        "REACH",
        "GRASP",
        "LIFT",
        "TRANSPORT",
        "PLACE"
    ],
    "GRASP": [
        "LOCATE",
        "APPROACH",
        "STABILIZE",
        "REACH",
        "GRASP",
        "LIFT"
    ],
    "NAVIGATE": [
        "LOCATE",
        "PLAN_PATH",
        "APPROACH",
        "STABILIZE"
    ],
    "STAND": [
        "STABILIZE",
        "MONITOR_BALANCE"
    ],
    "INSPECT": [
        "LOCATE",
        "PITCH_HEAD",
        "CAPTURE_FRAME"
    ],
    "WAVE": [
        "STABILIZE",
        "REACH",
        "OSCILLATE_WRIST",
        "RETRACT"
    ]
}


class NLPPipeline:
    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        """Clean and normalize raw text."""
        text = text.strip().lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_intent(self, text: str) -> Tuple[str, float]:
        """Classify instruction intent using rules and keyword semantics."""
        norm = self.normalize(text)
        if ("pick" in norm or "take" in norm or "grab" in norm) and ("place" in norm or "put" in norm):
            return "PICK_AND_PLACE", 1.0
        elif "pick" in norm or "grasp" in norm or "grab" in norm:
            return "GRASP", 0.95
        elif "walk" in norm or "go" in norm or "navigate" in norm or "approach" in norm:
            return "NAVIGATE", 0.90
        elif "stand" in norm or "balance" in norm or "hold" in norm:
            return "STAND", 0.95
        elif "look" in norm or "inspect" in norm or "see" in norm:
            return "INSPECT", 0.90
        elif "wave" in norm or "greet" in norm:
            return "WAVE", 0.95
        return "UNKNOWN", 0.0

    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """Extract target object and destination entities."""
        norm = self.normalize(text)
        target_obj = None
        destination = None

        # Resolve compound entities like "red cube"
        if "red cube" in norm or "red block" in norm:
            target_obj = "red_cube"
        elif "blue cube" in norm or "blue block" in norm:
            target_obj = "blue_cube"
        elif "cube" in norm:
            target_obj = "red_cube"
        elif "table" in norm and "to the table" in norm:
            target_obj = "table"

        # Resolve destinations
        if "shelf" in norm:
            destination = "shelf"
        elif "table" in norm and ("on the table" in norm or "onto the table" in norm):
            destination = "table"
        elif "bin" in norm or "box" in norm:
            destination = "bin"
        elif destination is None and target_obj is not None:
            destination = "shelf"  # Default canonical goal destination

        return {
            "object": target_obj,
            "destination": destination
        }

    def parse_instruction(self, text: str, world_objects: Optional[List[str]] = None) -> TaskPlan:
        """Parse natural language command into validated TaskPlan."""
        intent, conf = self.extract_intent(text)
        if intent == "UNKNOWN":
            return TaskPlan(
                instruction=text,
                task="REJECTED",
                status="FAILED",
                metadata={"error": "Unsupported instruction intent", "confidence": conf}
            )

        entities = self.extract_entities(text)
        obj = entities["object"] or "unspecified"
        dest = entities["destination"] or "unspecified"

        # Validate against known world objects if supplied
        if world_objects is not None and obj != "unspecified" and obj != "self":
            if obj not in world_objects:
                return TaskPlan(
                    instruction=text,
                    task="REJECTED",
                    status="FAILED",
                    metadata={"error": f"Object '{obj}' not present in world", "world_objects": world_objects}
                )

        steps = SKILL_RECIPES.get(intent, ["IDLE"])

        return TaskPlan(
            instruction=text,
            task=intent,
            target_object=obj,
            destination=dest,
            steps=steps,
            status="PENDING",
            metadata={"confidence": conf, "raw_entities": entities}
        )
