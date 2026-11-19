from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class WeldType(str, Enum):
    FILLET = "fillet"
    BUTT = "butt"
    SPOT = "spot"


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WeldingTask:
    task_id: str
    part: str
    material: str
    weld_type: WeldType
    target_points: Sequence[Point3D]
    robot_id: str


@dataclass(frozen=True)
class RobotProgramStep:
    step_id: str
    action: str
    target: Point3D | None = None
    speed: float | None = None


@dataclass(frozen=True)
class RobotProgram:
    program_id: str
    task_id: str
    steps: Sequence[RobotProgramStep]


@dataclass(frozen=True)
class ValidationResult:
    program_id: str
    is_valid: bool
    errors: Sequence[str]
    warnings: Sequence[str]