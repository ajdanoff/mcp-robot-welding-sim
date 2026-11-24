from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np


SUPPORTED_MOVE_TYPES = {"MOVJ", "MOVL"}
SUPPORTED_ARC_STATES = {"ARC_ON", "ARC_OFF", None}
SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}

MELFA_MOVE_COMMANDS = {
    "MOVJ": "Mov",
    "MOVL": "Mvs",
}


@dataclass(frozen=True)
class PostprocessorConfig:
    """Controller- and cell-specific postprocessor configuration.

    Arc commands and delays must be validated for the exact robot controller
    and welding power-source interface before production use.
    """

    override_percent: float = 100.0
    melfa_arc_start_cmd: str = "ArcStart 1"
    melfa_arc_end_cmd: str = "ArcEnd 1"
    melfa_arc_start_delay_s: float = 0.0
    melfa_arc_end_delay_s: float = 0.0
    gcode_arc_start_cmd: str = "M62 P1"
    gcode_arc_end_cmd: str = "M64 P1"

    def __post_init__(self) -> None:
        if not 0.0 < self.override_percent <= 100.0:
            raise ValueError(
                "override_percent must be greater than 0 and at most 100."
            )

        if self.melfa_arc_start_delay_s < 0.0:
            raise ValueError("melfa_arc_start_delay_s must be non-negative.")

        if self.melfa_arc_end_delay_s < 0.0:
            raise ValueError("melfa_arc_end_delay_s must be non-negative.")

        for field_name in (
            "melfa_arc_start_cmd",
            "melfa_arc_end_cmd",
            "gcode_arc_start_cmd",
            "gcode_arc_end_cmd",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class TrajectoryWaypoint:
    """A validated trajectory waypoint in the robot base frame."""

    label: str
    move_type: str
    xyz_mm: np.ndarray
    joints_deg: np.ndarray
    orientation: np.ndarray
    arc_state: str | None = None
    tcp_speed_mm_s: float | None = None
    joint_speed_percent: float | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("label must not be empty.")

        if self.move_type not in SUPPORTED_MOVE_TYPES:
            raise ValueError(
                f"Unsupported move type: '{self.move_type}'. "
                f"Supported types: {sorted(SUPPORTED_MOVE_TYPES)}."
            )

        if self.arc_state not in SUPPORTED_ARC_STATES:
            supported_states = sorted(str(s) for s in SUPPORTED_ARC_STATES)
            raise ValueError(
                f"Unsupported arc state: '{self.arc_state}'. "
                f"Supported states: {supported_states}."
            )
        
        xyz = _finite_vector(self.xyz_mm, "xyz_mm", (3,))
        joints = _finite_vector(self.joints_deg, "joints_deg", None)
        orientation = _finite_rotation(self.orientation, "orientation")

        if joints.ndim != 1 or joints.size == 0:
            raise ValueError(
                "joints_deg must be a non-empty 1D array."
            )

        if self.move_type == "MOVJ":
            if self.tcp_speed_mm_s is not None:
                raise ValueError(
                    "MOVJ waypoints must use joint_speed_percent, "
                    "not tcp_speed_mm_s."
                )

            if (
                self.joint_speed_percent is None
                or not 0.0 < self.joint_speed_percent <= 100.0
            ):
                raise ValueError(
                    "MOVJ waypoints require joint_speed_percent between "
                    "0 and 100."
                )

        if self.move_type == "MOVL":
            if self.joint_speed_percent is not None:
                raise ValueError(
                    "MOVL waypoints must use tcp_speed_mm_s, "
                    "not joint_speed_percent."
                )

            if (
                self.tcp_speed_mm_s is None
                or self.tcp_speed_mm_s <= 0.0
            ):
                raise ValueError(
                    "MOVL waypoints require a positive tcp_speed_mm_s."
                )

        object.__setattr__(self, "xyz_mm", xyz)
        object.__setattr__(self, "joints_deg", joints)
        object.__setattr__(self, "orientation", orientation)

        xyz.flags.writeable = False
        joints.flags.writeable = False
        orientation.flags.writeable = False


def _finite_vector(
    value: object,
    name: str,
    expected_shape: tuple[int, ...] | None,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)

    if expected_shape is not None and vector.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {vector.shape}."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")

    return np.array(vector, dtype=np.float64, copy=True)


def _finite_rotation(
    value: object,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError(
            f"{name} must be a 3x3 rotation matrix; got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")

    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be an orthonormal rotation matrix.")

    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation matrix.")

    return np.array(matrix, dtype=np.float64, copy=True)


def _validate_program_name(program_name: str) -> str:
    name = program_name.strip()

    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", name):
        raise ValueError(
            "Program name must contain 1-64 letters, digits, underscores, "
            "or hyphens."
        )

    return name


def _rotation_to_euler_zyx(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """
    Converts a rotation matrix to intrinsic Z-Y-X Euler angles in degrees.

    The result is (A, B, C), where A is rotation about Z, B about Y, and
    C about X. This convention must be verified against the target controller.
    """
    sy = -float(rotation[2, 0])

    if abs(sy) >= 1.0 - 1e-9:
        pitch = np.pi / 2.0 if sy > 0.0 else -np.pi / 2.0
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    else:
        pitch = float(np.arcsin(sy))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    return (
        float(np.rad2deg(yaw)),
        float(np.rad2deg(pitch)),
        float(np.rad2deg(roll)),
    )


def _rotation_to_melfa_abc_deg(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """
    Converts a 3x3 base-to-tool rotation matrix to MELFA fixed XYZ Euler
    angles (A, B, C) in degrees, where A=Rx, B=Ry, C=Rz.
    """
    sy = -float(rotation.item((2, 0)))
    sy = max(-1.0, min(1.0, sy))
    b_rad = float(np.arcsin(sy))

    if abs(sy) < 1.0 - 1e-9:
        a_rad = float(np.arctan2(rotation.item((2, 1)), rotation.item((2, 2))))
        c_rad = float(np.arctan2(rotation.item((1, 0)), rotation.item((0, 0))))
    else:
        a_rad = float(np.arctan2(-rotation.item((1, 2)), rotation.item((1, 1))))
        c_rad = 0.0

    a_deg = round(float(np.rad2deg(a_rad)), 6) + 0.0
    b_deg = round(float(np.rad2deg(b_rad)), 6) + 0.0
    c_deg = round(float(np.rad2deg(c_rad)), 6) + 0.0

    return (a_deg, b_deg, c_deg)


def _joint_comment(waypoint: TrajectoryWaypoint) -> str:
    joints = " ".join(
        f"J{index + 1}={value:.3f}"
        for index, value in enumerate(waypoint.joints_deg)
    )
    return f"; JOINTS {joints}"


def _compile_gcode(
    waypoints: list[TrajectoryWaypoint],
    program_name: str,
    robot_type: str,
    config: PostprocessorConfig,
) -> str:
    lines: list[str] = [
        f"; PROGRAM: {program_name}",
        f"; ROBOT MODEL: {robot_type}",
        "; GENERATED BY MCP-ROBOT-WELDING-SIM",
        "; WARNING: Validate controller mapping and welding I/O before use.",
        "G21 ; Millimetres",
        "G90 ; Absolute positioning",
        "G94 ; Feed per minute",
    ]

    for waypoint in waypoints:
        x, y, z = waypoint.xyz_mm
        roll, pitch, yaw = _rotation_to_melfa_abc_deg(waypoint.orientation)

        motion_command = "G00" if waypoint.move_type == "MOVJ" else "G01"

        if waypoint.tcp_speed_mm_s is not None:
            feed_mm_min = waypoint.tcp_speed_mm_s * 60.0
            feed_word = f" F{feed_mm_min:.1f}"
        else:
            feed_word = ""

        lines.append(
            f"{motion_command} "
            f"X{x:.3f} "
            f"Y{y:.3f} "
            f"Z{z:.3f}"
            f"{feed_word} "
            f"; {waypoint.label}"
        )
        lines.append(_joint_comment(waypoint))
        lines.append(
            f"; ORIENTATION_A_B_C_DEG "
            f"{roll:.3f} {pitch:.3f} {yaw:.3f}"
        )

        if waypoint.arc_state == "ARC_ON":
            lines.append(
                f"{config.gcode_arc_start_cmd} ; CONFIGURED ARC_START"
            )
        elif waypoint.arc_state == "ARC_OFF":
            lines.append(
                f"{config.gcode_arc_end_cmd} ; CONFIGURED ARC_END"
            )

    lines.append("M30 ; End of program")
    return "\n".join(lines)


def _compile_melfa(
    waypoints: list[TrajectoryWaypoint],
    program_name: str,
    robot_type: str,
    config: PostprocessorConfig,
) -> str:
    lines: list[str] = [
        f"' PROGRAM: {program_name}",
        f"' ROBOT MODEL: {robot_type}",
        "' GENERATED BY MCP-ROBOT-WELDING-SIM",
        "' WARNING: Validate controller syntax, tool frame, and welding I/O.",
        "Servo On",
        f"Ovrd {config.override_percent:g}",
    ]

    for waypoint_index, waypoint in enumerate(waypoints, start=1):
        x, y, z = waypoint.xyz_mm
        roll, pitch, yaw = _rotation_to_melfa_abc_deg(waypoint.orientation)

        lines.append(
            f"P{waypoint_index} = "
            f"({x:.3f}, {y:.3f}, {z:.3f}, "
            f"{roll:.3f}, {pitch:.3f}, {yaw:.3f})(6, 0)"
        )

    lines.append("")

    for waypoint_index, waypoint in enumerate(waypoints, start=1):
        melfa_move = MELFA_MOVE_COMMANDS[waypoint.move_type]

        if waypoint.move_type == "MOVJ":
            joint_override = (
                waypoint.joint_speed_percent
                if waypoint.joint_speed_percent is not None
                else config.override_percent
            )
            lines.append(f"JOvrd {joint_override:.1f}")
        else:
            if waypoint.tcp_speed_mm_s is None:
                raise ValueError(
                    f"Waypoint '{waypoint.label}' is MOVL but has no TCP speed."
                )

            lines.append(f"Spd {waypoint.tcp_speed_mm_s:.3f}")

        lines.append(
            f"{melfa_move} P{waypoint_index} ' {waypoint.label}"
        )

        if waypoint.arc_state == "ARC_ON":
            lines.append(
                f"{config.melfa_arc_start_cmd} ' CONFIGURED ARC_START"
            )

            if config.melfa_arc_start_delay_s > 0.0:
                lines.append(
                    f"Dly {config.melfa_arc_start_delay_s:.3f}"
                )

        elif waypoint.arc_state == "ARC_OFF":
            lines.append(
                f"{config.melfa_arc_end_cmd} ' CONFIGURED ARC_END"
            )

            if config.melfa_arc_end_delay_s > 0.0:
                lines.append(
                    f"Dly {config.melfa_arc_end_delay_s:.3f}"
                )

    lines.append("")
    lines.append("Servo Off")
    lines.append("End")
    return "\n".join(lines)


def compile_program(
    waypoints: list[TrajectoryWaypoint],
    dialect: str = "melfa",
    program_name: str = "WELD_JOB_01",
    robot_type: str = "industrial_6dof",
    config: PostprocessorConfig | None = None,
) -> str:
    """
    Compiles validated waypoints into robot-controller code.

    Supported dialects:
    - `melfa`: MELFA-BASIC-style output using `Mov` and `Mvs`.
    - `gcode`: generic metric G-code-style output.

    Both outputs are simulation/export templates and must be validated against
    the exact controller, welding power source, tool frame, and safety
    configuration before use.
    """
    if not waypoints:
        raise ValueError("At least one trajectory waypoint is required.")

    if config is None:
        config = PostprocessorConfig()

    normalized_dialect = dialect.strip().lower()

    if normalized_dialect not in SUPPORTED_PROGRAM_DIALECTS:
        raise ValueError(
            f"Unsupported dialect: '{dialect}'. Supported dialects: "
            f"{sorted(SUPPORTED_PROGRAM_DIALECTS)}."
        )

    validated_program_name = _validate_program_name(program_name)

    if normalized_dialect == "gcode":
        return _compile_gcode(
            waypoints=waypoints,
            program_name=validated_program_name,
            robot_type=robot_type,
            config=config,
        )

    return _compile_melfa(
        waypoints=waypoints,
        program_name=validated_program_name,
        robot_type=robot_type,
        config=config,
    )