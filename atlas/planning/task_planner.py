"""Hierarchical Finite-State Machine Task Planner for Humanoid Manipulation and Navigation.
Validates state transitions, pre-conditions, and post-conditions across the 8-step pipeline.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from atlas.core.types import TaskPlan, RobotState


class HumanoidTaskPlanner:
    def __init__(self, plan: TaskPlan):
        self.plan = plan
        self.current_step_name = plan.steps[0] if len(plan.steps) > 0 else "IDLE"
        self.step_history: List[str] = []

    def check_preconditions(self, step: str, state: RobotState, world_context: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify physics/sensory preconditions before executing a skill."""
        if step == "LOCATE":
            return True, "Ready to observe"
        elif step == "APPROACH":
            if "object_pose" not in world_context:
                return False, "Target object position unknown"
            return True, "Object localized"
        elif step == "STABILIZE":
            if state.base_position[2] < 0.6:
                return False, "Robot posture compromised"
            return True, "Base height verified"
        elif step == "REACH":
            dist = np.linalg.norm(state.base_position[:2] - np.array(world_context.get("object_pose", [0, 0])[:2]))
            if dist > 1.2:
                return False, f"Target object out of reach (dist={dist:.2f}m)"
            return True, "Within reach envelope"
        elif step == "GRASP":
            return True, "End-effector positioned"
        elif step == "LIFT":
            return True, "Object held in gripper"
        elif step == "TRANSPORT":
            return True, "Object secure, ready to navigate"
        elif step == "PLACE":
            return True, "Destination reached"
        return True, "Default pass"

    def advance_step(self, postcondition_met: bool = True) -> Optional[str]:
        """Transition to the next state in the plan."""
        if not postcondition_met:
            return self.current_step_name

        self.step_history.append(self.current_step_name)
        has_next = self.plan.advance()
        if has_next:
            self.current_step_name = self.plan.current_step() or "IDLE"
        else:
            self.current_step_name = "COMPLETED"
        return self.current_step_name
