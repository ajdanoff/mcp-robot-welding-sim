from __future__ import annotations

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_robot_welding.kinematics import SixDofArm, create_manipulator
from mcp_robot_welding.mcp_server import (
    export_robot_program,
    inspect_workpiece,
    list_supported_robots,
    plan_and_simulate_weld,
)
from mcp_robot_welding.postprocessor import (
    PostprocessorConfig,
    TrajectoryWaypoint,
    compile_program,
)


def _make_waypoint(
    label: str = "TEST_PT",
    move_type: str = "MOVL",
    arc_state: str | None = None,
    dof: int = 6,
) -> TrajectoryWaypoint:
    return TrajectoryWaypoint(
        label=label,
        move_type=move_type,
        xyz_mm=np.array([500.0, 0.0, 200.0], dtype=np.float64),
        joints_deg=np.zeros(dof, dtype=np.float64),
        orientation=np.eye(3, dtype=np.float64),
        arc_state=arc_state,
        tcp_speed_mm_s=30.0 if move_type == "MOVL" else None,
        joint_speed_percent=40.0 if move_type == "MOVJ" else None,
    )


def test_inverse_kinematics_converges_to_fk_target() -> None:
    arm = SixDofArm()
    joints = np.array(
        [0.2, 0.25, 0.4, -0.15, 0.6, 0.1],
        dtype=np.float64,
    )
    target_pose = arm.forward_kinematics(joints)

    success, solution, position_error = arm.inverse_kinematics(
        target_pos=target_pose[:3, 3],
        target_rot=target_pose[:3, :3],
    )

    assert success
    assert position_error < 0.1

    solved_pose = arm.forward_kinematics(solution)
    assert np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3]) < 0.1


def test_compile_gcode_program() -> None:
    waypoints = [
        _make_waypoint("HOME", "MOVJ", None),
        _make_waypoint("SEAM_START", "MOVL", "ARC_ON"),
        _make_waypoint("SEAM_END", "MOVL", "ARC_OFF"),
    ]

    program = compile_program(
        waypoints=waypoints,
        dialect="gcode",
        program_name="TEST_PROGRAM",
        robot_type="industrial_6dof",
        config=PostprocessorConfig(
            gcode_arc_start_cmd="M62 P1",
            gcode_arc_end_cmd="M64 P1",
        ),
    )

    assert "G21" in program
    assert "G90" in program
    assert "G94" in program
    assert "M62 P1" in program
    assert "M64 P1" in program
    assert "M30" in program


def test_unsupported_dialect_raises_error() -> None:
    waypoint = _make_waypoint()

    with pytest.raises(ValueError):
        compile_program(
            waypoints=[waypoint],
            dialect="kuka",
        )


def test_invalid_move_type_raises_error() -> None:
    with pytest.raises(ValueError):
        _make_waypoint(move_type="MOVC")


def test_invalid_arc_state_raises_error() -> None:
    with pytest.raises(ValueError):
        _make_waypoint(arc_state="ARC_START")


def test_full_mcp_welding_pipeline() -> None:
    workpiece_info = inspect_workpiece()

    assert workpiece_info["part_name"] == "Bracket_Assembly_REV_B"
    assert len(workpiece_info["available_seams"]) == 3

    planning_result = plan_and_simulate_weld(
        seam_ids=["seam_top_rib", "seam_left_flange"],
        weld_speed_mm_s=30.0,
        robot_type="industrial_6dof",
    )

    assert (
        planning_result["status"] == "VALIDATED_OK"
    ), f"Unexpected planning response: {planning_result}"
    assert "plan_id" in planning_result
    assert planning_result["seams_planned"] == [
        "seam_top_rib",
        "seam_left_flange",
    ]

    waypoint_labels = [
        waypoint["label"] for waypoint in planning_result["waypoints_summary"]
    ]

    assert waypoint_labels[0] == "HOME_SAFE"
    assert waypoint_labels[-1] == "RETURN_HOME"
    assert "seam_top_rib_START" in waypoint_labels
    assert "seam_top_rib_SAFE_TRANSIT" in waypoint_labels
    assert "seam_left_flange_END" in waypoint_labels
    assert "seam_left_flange_SAFE_TRANSIT" in waypoint_labels
    assert "FINAL_SAFE_TRANSIT" in waypoint_labels

    plan_id = planning_result["plan_id"]

    compilation_result = export_robot_program(
        plan_id=plan_id,
        dialect="melfa",
        program_name="BRACKET_WELD_01",
    )

    assert compilation_result["status"] == "COMPILED"
    assert compilation_result["dialect"] == "melfa"
    assert compilation_result["robot_type"] == "industrial_6dof"

    program_code = compilation_result["program_code"]

    assert "' PROGRAM: BRACKET_WELD_01" in program_code
    assert "' ROBOT MODEL: industrial_6dof" in program_code
    assert "Servo On" in program_code
    assert "ArcStart 1" in program_code
    assert "ArcEnd 1" in program_code
    assert "Mov P1" in program_code
    assert "Mvs " in program_code
    assert "End" in program_code


def test_unknown_seam_is_rejected() -> None:
    with pytest.raises(ToolError, match="Unknown seam_id"):
        plan_and_simulate_weld(
            seam_ids=["non_existent_seam"],
        )


def test_unknown_robot_type_is_rejected() -> None:
    with pytest.raises(ToolError, match="Unknown robot_type"):
        plan_and_simulate_weld(
            seam_ids=["seam_top_rib"],
            robot_type="delta_3dof",
        )


def test_export_requires_plan_id() -> None:
    with pytest.raises(Exception, match="Unknown or expired plan_id"):
        export_robot_program(
            plan_id="does_not_exist",
            dialect="melfa",
            program_name="SHOULD_NOT_COMPILE",
        )


def test_multiple_robot_kinematic_profiles() -> None:
    robots_info = list_supported_robots()
    robot_types = {robot["robot_type"] for robot in robots_info["supported_robots"]}

    assert robot_types == {
        "industrial_6dof",
        "cobot_6dof",
        "scara_4dof",
    }

    cobot_plan = plan_and_simulate_weld(
        seam_ids=["seam_top_rib"],
        weld_speed_mm_s=25.0,
        robot_type="cobot_6dof",
    )

    assert cobot_plan["status"] == "VALIDATED_OK", f"Cobot failed: {cobot_plan}"
    assert cobot_plan["dof"] == 6

    scara_plan = plan_and_simulate_weld(
        seam_ids=["seam_top_rib"],
        weld_speed_mm_s=40.0,
        robot_type="scara_4dof",
    )

    assert scara_plan["status"] == "VALIDATED_OK", f"SCARA failed: {scara_plan}"
    assert scara_plan["dof"] == 4

    compiled_scara = export_robot_program(
        plan_id=scara_plan["plan_id"],
        dialect="gcode",
        program_name="SCARA_SPOT_01",
    )

    assert compiled_scara["status"] == "COMPILED"
    assert compiled_scara["robot_type"] == "scara_4dof"
    assert "; ROBOT MODEL: scara_4dof" in compiled_scara["program_code"]


def test_tool_orientation_reachable_for_top_seam():
    arm = create_manipulator("industrial_6dof")

    seam_normal = np.array([0.0, 0.0, 1.0])
    seam_tangent = np.array([1.0, 0.0, 0.0])
    tool_z_direction = -seam_normal

    target_rotation = arm.make_tool_orientation(
        tool_z_direction,
        reference_x_axis=seam_tangent,
    )

    target_position = np.array([480.0, 0.0, 180.0])

    success, q, position_error = arm.inverse_kinematics(
        target_pos=target_position,
        target_rot=target_rotation,
        pos_tol=0.1,
        rot_tol=0.015,
    )

    assert success, f"IK failed: position_error={position_error}"

    pose = arm.forward_kinematics(q)
    orientation_error = arm._rotation_error(
        pose[:3, :3],
        target_rotation,
    )

    assert np.linalg.norm(orientation_error) < 0.015
