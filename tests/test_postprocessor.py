import numpy as np

import pytest

from mcp_robot_welding.postprocessor import (
    TrajectoryWaypoint,
    compile_program,
)


def _make_waypoint(
    label: str = "TEST_POINT",
    move_type: str = "MOVJ",
    arc_state: str | None = None,
) -> TrajectoryWaypoint:
    return TrajectoryWaypoint(
        label=label,
        move_type=move_type,
        xyz_mm=np.array([100.0, 200.0, 300.0]),
        joints_deg=np.zeros(6),
        arc_state=arc_state,
        speed_mm_s=25.0,
    )


def test_compile_melfa_program():
    waypoint = _make_waypoint()

    program = compile_program(
        waypoints=[waypoint],
        dialect="melfa",
        program_name="TEST_PROGRAM",
    )

    assert "PROGRAM: TEST_PROGRAM" in program
    assert "SERVO ON" in program
    assert "MOVJ P1" in program
    assert "SERVO OFF" in program
    assert "END" in program


def test_compile_gcode_program():
    waypoint = _make_waypoint(
        move_type="MOVL",
        arc_state="ARC_ON",
    )

    program = compile_program(
        waypoints=[waypoint],
        dialect="gcode",
        program_name="TEST_PROGRAM",
    )

    assert "G21" in program
    assert "G90" in program
    assert "G01 X100.000" in program
    assert "M03 S100" in program
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