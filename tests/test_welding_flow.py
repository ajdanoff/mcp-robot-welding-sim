from mcp_robot_welding.mcp_server import (
    export_robot_program,
    inspect_workpiece,
    plan_and_simulate_weld,
)

import numpy as np

from mcp_robot_welding.kinematics import SixDofArm


def test_inverse_kinematics_converges_to_fk_target() -> None:
    arm = SixDofArm()

    joints = np.array(
        [
            0.2,
            -0.4,
            0.6,
            0.3,
            -0.5,
            0.1,
        ]
    )

    target_pose = arm.forward_kinematics(joints)

    success, solution, position_error = arm.inverse_kinematics(
        target_pos=target_pose[:3, 3],
        target_rot=target_pose[:3, :3],
    )

    assert success
    assert position_error < 0.1

    solved_pose = arm.forward_kinematics(solution)

    assert np.linalg.norm(
        solved_pose[:3, 3] - target_pose[:3, 3]
    ) < 0.1

    orientation_error = arm._rotation_error(
        solved_pose[:3, :3],
        target_pose[:3, :3],
    )

    assert np.linalg.norm(orientation_error) < 0.01


def test_full_mcp_welding_pipeline() -> None:
    # 1. Agent inspects loaded CAD part.
    workpiece_info = inspect_workpiece()

    assert workpiece_info["part_name"] == "Bracket_Assembly_REV_B"
    assert len(workpiece_info["available_seams"]) == 3

    # 2. Agent plans welding for two seams requested by the operator.
    planning_result = plan_and_simulate_weld(
        seam_ids=[
            "seam_top_rib",
            "seam_left_flange",
        ],
        weld_speed_mm_s=30.0,
    )

    assert planning_result["status"] == "VALIDATED_OK", (
        f"Unexpected planning response: {planning_result}"
    )

    assert planning_result["collision_free"] is True
    assert planning_result["seams_planned"] == [
        "seam_top_rib",
        "seam_left_flange",
    ]

    assert planning_result["total_waypoints"] == 12

    waypoint_labels = [
        waypoint["label"]
        for waypoint in planning_result["waypoints_summary"]
    ]

    assert waypoint_labels[0] == "HOME_SAFE"
    assert waypoint_labels[-1] == "RETURN_HOME"

    assert "seam_top_rib_START" in waypoint_labels
    assert "seam_top_rib_END" in waypoint_labels

    assert "seam_left_flange_START" in waypoint_labels
    assert "seam_left_flange_END" in waypoint_labels

    # 3. Postprocessor compiles a MELFA-BASIC robot program.
    compilation_result = export_robot_program(
        dialect="melfa",
        program_name="BRACKET_WELD_01",
    )

    assert compilation_result["status"] == "COMPILED"
    assert compilation_result["dialect"] == "melfa"

    program_code = compilation_result["program_code"]

    assert "PROGRAM: BRACKET_WELD_01" in program_code
    assert "SERVO ON" in program_code
    assert "SERVO OFF" in program_code
    assert "END" in program_code

    assert "HOPEN 1" in program_code
    assert "HCLOSE 1" in program_code

    assert "MOVL" in program_code
    assert "MOVJ" in program_code

    # 4. Safety check: unknown seam is rejected before motion generation.
    invalid_planning_result = plan_and_simulate_weld(
        seam_ids=["non_existent_seam"],
    )

    assert invalid_planning_result["status"] == "ERROR"
    assert "Unknown seam_id" in invalid_planning_result["reason"]

