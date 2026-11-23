from __future__ import annotations

from dataclasses import dataclass
import math
import re
import numpy as np

SUPPORTED_MOVE_TYPES = {"MOVJ", "MOVL"}
SUPPORTED_ARC_STATES = {"ARC_ON", "ARC_OFF", None}
SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}

MELFA_MOVE_COMMANDS = {
    "MOVJ": "Mov",
    "MOVL": "Mvs",
}

_PROGRAM_NAME_PATTERN = re.compile(r"[A-Za-z0-9_\-]{1,64}")


@dataclass(frozen=True)
class PostprocessorConfig:
    """Integrator-validated settings for target controller export."""

    override_percent: int = 100
    melfa_arc_start_cmd: str = "ArcStart 1"
    melfa_arc_end_cmd: str = "ArcEnd 1"
    melfa_arc_start_delay_s: float | None = None
    melfa_arc_end_delay_s: float | None = None
    melfa_posture_flags: str = "(6, 0)"
    gcode_arc_start_cmd: str = "M62 P1 ; CONFIGURE WELD INTERFACE"
    gcode_arc_end_cmd: str = "M64 P1 ; CONFIGURE WELD INTERFACE"

    def __post_init__(self) -> None:
        if not (1 <= self.override_percent <= 100):
            raise ValueError("override_percent must be between 1 and 100.")
        if self.melfa_arc_start_delay_s is not None and (
            not math.isfinite(self.melfa_arc_start_delay_s)
            or self.melfa_arc_start_delay_s < 0
        ):
            raise ValueError("melfa_arc_start_delay_s must be non-negative and finite.")
        if self.melfa_arc_end_delay_s is not None and (
            not math.isfinite(self.melfa_arc_end_delay_s)
            or self.melfa_arc_end_delay_s < 0
        ):
            raise ValueError("melfa_arc_end_delay_s must be non-negative and finite.")


@dataclass(frozen=True)
class TrajectoryWaypoint:
    label: str
    move_type: str
    xyz_mm: np.ndarray
    joints_deg: np.ndarray
    orientation: np.ndarray
    arc_state: str | None = None
    tcp_speed_mm_s: float | None = None
    joint_speed_percent: float | None = None

    def __post_init__(self) -> None:
        if self.move_type not in SUPPORTED_MOVE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_MOVE_TYPES))
            raise ValueError(
                f"Unsupported move type: '{self.move_type}'. Supported types: {supported}."
            )

        if self.arc_state not in SUPPORTED_ARC_STATES:
            supported_states = ", ".join(str(s) for s in SUPPORTED_ARC_STATES)
            raise ValueError(
                f"Unsupported arc state: '{self.arc_state}'. Supported states: {supported_states}."
            )

        if self.tcp_speed_mm_s is not None:
            if not math.isfinite(self.tcp_speed_mm_s) or self.tcp_speed_mm_s <= 0:
                raise ValueError("tcp_speed_mm_s must be a positive finite number.")

        if self.joint_speed_percent is not None:
            if (
                not math.isfinite(self.joint_speed_percent)
                or not (0 < self.joint_speed_percent <= 100)
            ):
                raise ValueError("joint_speed_percent must be in range (0, 100].")

        xyz = np.array(self.xyz_mm, dtype=np.float64, copy=True)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("xyz_mm must contain 3 finite coordinates.")

        joints = np.array(self.joints_deg, dtype=np.float64, copy=True)
        if joints.ndim != 1 or joints.size == 0 or not np.all(np.isfinite(joints)):
            raise ValueError("joints_deg must be a non-empty 1D array of finite values.")

        rot = np.array(self.orientation, dtype=np.float64, copy=True)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError("orientation must be a 3x3 rotation matrix of finite values.")

        xyz.flags.writeable = False
        joints.flags.writeable = False
        rot.flags.writeable = False

        object.__setattr__(self, "xyz_mm", xyz)
        object.__setattr__(self, "joints_deg", joints)
        object.__setattr__(self, "orientation", rot)

def _rotation_matrix_to_melfa_abc_deg(rot: np.ndarray) -> tuple[float, float, float]:
    """
    Converts a 3x3 base-to-tool rotation matrix to MELFA fixed XYZ Euler angles (A, B, C) in degrees.
    Convention: R = Rz(C) * Ry(B) * Rx(A).
    """
    sy = -float(rot.item((2, 0)))
    sy = max(-1.0, min(1.0, sy))
    b_rad = math.asin(sy)

    if abs(sy) < 1.0 - 1e-9:
        a_rad = math.atan2(float(rot.item((2, 1))), float(rot.item((2, 2))))
        c_rad = math.atan2(float(rot.item((1, 0))), float(rot.item((0, 0))))
    else:
        # Gimbal lock handling
        a_rad = math.atan2(-float(rot.item((1, 2))), float(rot.item((1, 1))))
        c_rad = 0.0

    # Normalize -0.0 to 0.0 so formatted controller code never emits '-0.000'
    a_deg = round(math.degrees(a_rad), 6) + 0.0
    b_deg = round(math.degrees(b_rad), 6) + 0.0
    c_deg = round(math.degrees(c_rad), 6) + 0.0

    return (a_deg, b_deg, c_deg)

def _validate_program_name(program_name: str) -> str:
    name = program_name.strip()
    if not _PROGRAM_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Program name must contain 1-64 letters, digits, underscores, or hyphens."
        )
    return name


def _sanitize_comment(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def _compile_gcode(
    waypoints: list[TrajectoryWaypoint],
    program_name: str,
    robot_type: str,
    config: PostprocessorConfig,
) -> str:
    lines: list[str] = [
        f"; PROGRAM: {program_name}",
        f"; ROBOT MODEL: {_sanitize_comment(robot_type)}",
        "; GENERATED BY MCP-ROBOT-WELDING-SIM",
        "; WARNING: Validate postprocessor G-code dialect and weld I/O before execution.",
        "G21 ; Millimeters",
        "G90 ; Absolute positioning",
        "G94 ; Feed per minute",
    ]

    for waypoint in waypoints:
        x, y, z = waypoint.xyz_mm
        label = _sanitize_comment(waypoint.label)

        joint_text = " ".join(
            f"J{i + 1}={val:.3f}" for i, val in enumerate(waypoint.joints_deg)
        )
        lines.append(f"; JOINTS {joint_text}")

        if waypoint.move_type == "MOVJ":
            lines.append(f"G00 X{x:.3f} Y{y:.3f} Z{z:.3f} ; {label}")
        else:
            speed_mm_s = (
                waypoint.tcp_speed_mm_s if waypoint.tcp_speed_mm_s is not None else 50.0
            )
            feed_mm_min = speed_mm_s * 60.0
            lines.append(
                f"G01 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_mm_min:.1f} ; {label}"
            )

        if waypoint.arc_state == "ARC_ON":
            lines.append(config.gcode_arc_start_cmd)
        elif waypoint.arc_state == "ARC_OFF":
            lines.append(config.gcode_arc_end_cmd)

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
        f"' ROBOT MODEL: {_sanitize_comment(robot_type)}",
        "' GENERATED BY MCP-ROBOT-WELDING-SIM (MELFA-BASIC TEMPLATE)",
        "' WARNING: Validate tool frame, posture flags, and weld I/O before production.",
        "Servo On",
        f"Ovrd {config.override_percent}",
        "",
    ]

    for idx, waypoint in enumerate(waypoints, start=1):
        x, y, z = waypoint.xyz_mm
        a_deg, b_deg, c_deg = _rotation_matrix_to_melfa_abc_deg(waypoint.orientation)
        lines.append(
            f"P{idx} = ({x:.3f}, {y:.3f}, {z:.3f}, "
            f"{a_deg:.3f}, {b_deg:.3f}, {c_deg:.3f}){config.melfa_posture_flags}"
        )

    lines.append("")

    for idx, waypoint in enumerate(waypoints, start=1):
        label = _sanitize_comment(waypoint.label)
        melfa_cmd = MELFA_MOVE_COMMANDS[waypoint.move_type]

        if waypoint.move_type == "MOVJ" and waypoint.joint_speed_percent is not None:
            lines.append(f"JOvrd {waypoint.joint_speed_percent:.1f}")
        elif waypoint.move_type == "MOVL" and waypoint.tcp_speed_mm_s is not None:
            lines.append(f"Spd {waypoint.tcp_speed_mm_s:.3f}")

        lines.append(f"{melfa_cmd} P{idx} ' {label}")

        if waypoint.arc_state == "ARC_ON":
            lines.append(config.melfa_arc_start_cmd)
            if config.melfa_arc_start_delay_s is not None:
                lines.append(f"Dly {config.melfa_arc_start_delay_s:.3f}")
        elif waypoint.arc_state == "ARC_OFF":
            lines.append(config.melfa_arc_end_cmd)
            if config.melfa_arc_end_delay_s is not None:
                lines.append(f"Dly {config.melfa_arc_end_delay_s:.3f}")

    lines.extend(["", "Servo Off", "End"])
    return "\n".join(lines)


def compile_program(
    waypoints: list[TrajectoryWaypoint],
    dialect: str = "melfa",
    program_name: str = "WELD_JOB_01",
    robot_type: str = "industrial_6dof",
    config: PostprocessorConfig | None = None,
) -> str:
    """
    Compiles validated waypoints into target industrial robot controller code.

    Supported dialects:
    - 'melfa': MELFA-BASIC template program (requires integrator validation).
    - 'gcode': ISO G-code template program (requires postprocessor validation).
    """
    if not waypoints:
        raise ValueError("At least one trajectory waypoint is required.")

    normalized_dialect = dialect.strip().lower()
    if normalized_dialect not in SUPPORTED_PROGRAM_DIALECTS:
        raise ValueError(
            f"Unsupported dialect: '{dialect}'. "
            f"Supported dialects: {sorted(SUPPORTED_PROGRAM_DIALECTS)}."
        )

    validated_program_name = _validate_program_name(program_name)
    active_config = config if config is not None else PostprocessorConfig()

    if normalized_dialect == "gcode":
        return _compile_gcode(
            waypoints=waypoints,
            program_name=validated_program_name,
            robot_type=robot_type,
            config=active_config,
        )

    return _compile_melfa(
        waypoints=waypoints,
        program_name=validated_program_name,
        robot_type=robot_type,
        config=active_config,
    )
