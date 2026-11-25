import numpy as np
import pytest

from mcp_robot_welding.postprocessor import (
    PostprocessorConfig,
    TrajectoryWaypoint,
    compile_program,
)


def _make_waypoint(
    label: str = "TEST_POINT",
    move_type: str = "MOVJ",
    arc_state: str | None = None,
    orientation: np.ndarray | None = None,
    tcp_speed_mm_s: float | None = None,
    joint_speed_percent: float | None = None,
) -> TrajectoryWaypoint:
    rot = np.eye(3) if orientation is None else orientation
    return TrajectoryWaypoint(
        label=label,
        move_type=move_type,
        xyz_mm=np.array([100.0, 200.0, 300.0]),
        joints_deg=np.zeros(6),
        orientation=rot,
        arc_state=arc_state,
        tcp_speed_mm_s=25.0 if move_type == "MOVL" else None,
        joint_speed_percent=40.0 if move_type == "MOVJ" else None,
    )


def test_compile_melfa_program():
    # 180-deg rotation around X axis -> A=180.000, B=0.000, C=0.000
    rot_x_180 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    waypoints = [
        _make_waypoint(label="HOME", move_type="MOVJ", orientation=rot_x_180),
        _make_waypoint(
            label="WELD_START",
            move_type="MOVL",
            arc_state="ARC_ON",
            orientation=rot_x_180,
        ),
        _make_waypoint(
            label="WELD_END",
            move_type="MOVL",
            arc_state="ARC_OFF",
            orientation=rot_x_180,
        ),
    ]

    program = compile_program(
        waypoints=waypoints,
        dialect="melfa",
        program_name="TEST_PROGRAM",
    )

    assert "PROGRAM: TEST_PROGRAM" in program
    assert "Servo On" in program
    assert "Ovrd 100" in program
    assert "P1 = (100.000, 200.000, 300.000, 180.000, 0.000, 0.000)(6, 0)" in program
    assert "JOvrd 40.0" in program
    assert "Mov P1" in program
    assert "Spd 25.000" in program
    assert "Mvs P2" in program
    assert "ArcStart 1" in program
    assert "ArcEnd 1" in program
    assert "HOPEN" not in program
    assert "HCLOSE" not in program
    assert "Servo Off" in program
    assert program.splitlines()[-1] == "End"


def test_compile_melfa_custom_weld_config():
    config = PostprocessorConfig(
        override_percent=80,
        melfa_arc_start_cmd="M_Out(1)=1",
        melfa_arc_end_cmd="M_Out(1)=0",
        melfa_arc_start_delay_s=0.25,
    )
    waypoint = _make_waypoint(move_type="MOVL", arc_state="ARC_ON")

    program = compile_program(
        waypoints=[waypoint],
        dialect="melfa",
        program_name="CUSTOM_CFG",
        config=config,
    )

    assert "Ovrd 80" in program
    assert "M_Out(1)=1" in program
    assert "Dly 0.250" in program


def test_compile_gcode_program():
    waypoints = [
        _make_waypoint(label="APPROACH", move_type="MOVJ"),
        _make_waypoint(label="WELD", move_type="MOVL", arc_state="ARC_ON"),
        _make_waypoint(label="STOP", move_type="MOVL", arc_state="ARC_OFF"),
    ]

    program = compile_program(
        waypoints=waypoints,
        dialect="gcode",
        program_name="TEST_PROGRAM",
    )

    assert "G21" in program
    assert "G90" in program
    assert "G94" in program
    assert "; JOINTS J1=0.000" in program
    assert "G00 X100.000 Y200.000 Z300.000" in program
    assert "G01 X100.000 Y200.000 Z300.000 F1500.0" in program
    assert "M62 P1" in program
    assert "M64 P1" in program
    assert "M03" not in program
    assert "M30" in program


def test_unsupported_dialect_raises_error():
    waypoint = _make_waypoint()

    with pytest.raises(ValueError):
        compile_program(
            waypoints=[waypoint],
            dialect="kuka",
        )


def test_invalid_move_type_raises_error():
    with pytest.raises(ValueError):
        _make_waypoint(move_type="MOVEJ")


def test_invalid_arc_state_raises_error():
    with pytest.raises(ValueError):
        _make_waypoint(arc_state="ARC_START")


def test_non_finite_coordinates_raise_error():
    with pytest.raises(ValueError):
        TrajectoryWaypoint(
            label="NAN_PT",
            move_type="MOVL",
            xyz_mm=np.array([100.0, np.nan, 300.0]),
            joints_deg=np.zeros(6),
            orientation=np.eye(3),
            tcp_speed_mm_s=25.0,
        )


def test_program_name_injection_rejected():
    waypoint = _make_waypoint()
    with pytest.raises(ValueError):
        compile_program(
            waypoints=[waypoint],
            dialect="melfa",
            program_name="BAD_NAME\nServo Off",
        )


def test_waypoint_arrays_are_immutable():
    waypoint = _make_waypoint()
    with pytest.raises(ValueError):
        waypoint.xyz_mm[:] = 999.0


def test_movj_rejects_tcp_speed():
    with pytest.raises(ValueError):
        TrajectoryWaypoint(
            label="BAD_MOVJ",
            move_type="MOVJ",
            xyz_mm=np.array([100.0, 200.0, 300.0]),
            joints_deg=np.zeros(6),
            orientation=np.eye(3),
            tcp_speed_mm_s=25.0,
        )
