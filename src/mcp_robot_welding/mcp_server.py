from __future__ import annotations

import json
import numpy as np
from mcp.server.mcpserver import MCPServer

from .kinematics import SUPPORTED_ROBOTS, create_manipulator
from .postprocessor import TrajectoryWaypoint, compile_program
from .scene import WorkpieceScene

mcp = MCPServer("Industrial-Robot-Welding-Simulator")

scene = WorkpieceScene()
last_validated_plan: list[TrajectoryWaypoint] = []
last_robot_type: str = "industrial_6dof"

SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}
MAX_JOINT_JUMP_DEGREES = 120.0
WELD_PATH_COLLISION_SAMPLES = 20


@mcp.resource("scene://workpiece/seams")
def get_workpiece_seams_resource() -> str:
    """Provides CAD workpiece metadata and available weld seams as JSON."""
    payload = {
        "part_name": scene.part_name,
        "seams": [seam.model_dump() for seam in scene.seams.values()],
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def list_supported_robots() -> dict:
    """Lists available manipulator kinematic models and their degrees of freedom."""
    models = []
    for r_type, desc in SUPPORTED_ROBOTS.items():
        manipulator = create_manipulator(r_type)
        models.append(
            {
                "robot_type": r_type,
                "dof": manipulator.dof,
                "max_reach_mm": manipulator.max_reach_mm,
                "description": desc,
            }
        )
    return {"supported_robots": models}


@mcp.tool()
def inspect_workpiece() -> dict:
    """Returns the loaded 3D workpiece name, bounding obstacles, and available weld seam IDs."""
    return {
        "part_name": scene.part_name,
        "available_seams": [
            {
                "seam_id": seam.seam_id,
                "description": seam.description,
                "start_xyz": seam.start_xyz,
                "end_xyz": seam.end_xyz,
            }
            for seam in scene.seams.values()
        ],
    }


@mcp.tool()
def plan_and_simulate_weld(
    seam_ids: list[str],
    weld_speed_mm_s: float = 25.0,
    approach_clearance_mm: float = 40.0,
    robot_type: str = "industrial_6dof",
) -> dict:
    """
    Plans approach, linear weld, and retract trajectories for requested seams
    using the selected manipulator kinematic profile ('industrial_6dof', 'cobot_6dof', 'scara_4dof').
    """
    global last_validated_plan, last_robot_type

    if not seam_ids:
        return {"status": "ERROR", "reason": "At least one seam_id is required."}
    if weld_speed_mm_s <= 0:
        return {"status": "ERROR", "reason": "weld_speed_mm_s must be positive."}
    if approach_clearance_mm < 0:
        return {"status": "ERROR", "reason": "approach_clearance_mm must be non-negative."}

    try:
        arm = create_manipulator(robot_type)
    except ValueError as exc:
        return {"status": "ERROR", "reason": str(exc)}

    planned_waypoints: list[TrajectoryWaypoint] = []
    home_xyz = np.array([450.0, 0.0, 420.0], dtype=np.float64)
    home_tool_orientation = arm.make_tool_orientation(np.array([0.0, 0.0, -1.0], dtype=np.float64))

    home_ik_success, q_home, _ = arm.inverse_kinematics(
        target_pos=home_xyz,
        target_rot=home_tool_orientation,
    )
    if not home_ik_success:
        return {"status": "ERROR", "reason": f"Failed to solve IK for HOME pose on '{arm.robot_type}'."}

    planned_waypoints.append(
        TrajectoryWaypoint(
            label="HOME_SAFE",
            move_type="MOVJ",
            xyz_mm=home_xyz,
            joints_deg=np.rad2deg(q_home),
            speed_mm_s=150.0,
        )
    )
    q_seed = q_home.copy()

    for seam_id in seam_ids:
        if seam_id not in scene.seams:
            return {
                "status": "ERROR",
                "reason": f"Unknown seam_id '{seam_id}'. Available seam IDs: {list(scene.seams.keys())}.",
            }

        seam = scene.seams[seam_id]
        seam_start = np.asarray(seam.start_xyz, dtype=np.float64)
        seam_end = np.asarray(seam.end_xyz, dtype=np.float64)
        seam_normal = np.asarray(seam.normal_xyz, dtype=np.float64)
        seam_normal = seam_normal / np.linalg.norm(seam_normal)

        approach_xyz = seam_start + seam_normal * approach_clearance_mm
        retract_xyz = seam_end + seam_normal * approach_clearance_mm
        tool_orientation = arm.make_tool_orientation(-seam_normal)
        weld_path_samples = np.linspace(seam_start, seam_end, WELD_PATH_COLLISION_SAMPLES)

        weld_steps = [
            (f"{seam_id}_APPROACH", "MOVJ", approach_xyz, None, 120.0),
            (f"{seam_id}_START", "MOVL", seam_start, "ARC_ON", weld_speed_mm_s),
            (f"{seam_id}_MID", "MOVL", 0.5 * (seam_start + seam_end), None, weld_speed_mm_s),
            (f"{seam_id}_END", "MOVL", seam_end, "ARC_OFF", weld_speed_mm_s),
            (f"{seam_id}_RETRACT", "MOVJ", retract_xyz, None, 120.0),
        ]

        for waypoint_label, move_type, target_xyz, arc_command, waypoint_speed in weld_steps:
            ik_success, q_solution, position_error = arm.inverse_kinematics(
                target_pos=target_xyz,
                target_rot=tool_orientation,
                q_init=q_seed,
            )
            if not ik_success:
                return {
                    "status": "IK_UNREACHABLE",
                    "robot_type": arm.robot_type,
                    "waypoint": waypoint_label,
                    "target_xyz": target_xyz.tolist(),
                    "position_error_mm": round(position_error, 3),
                }

            joint_jump_deg = arm._normalized_joint_jump(q_solution, q_seed)
            if joint_jump_deg > MAX_JOINT_JUMP_DEGREES:
                return {
                    "status": "JOINT_DISCONTINUITY",
                    "robot_type": arm.robot_type,
                    "waypoint": waypoint_label,
                    "details": f"Large joint-space jump: {joint_jump_deg:.1f} degrees.",
                }

            waypoint_frames = arm.forward_kinematics_chain(q_solution)
            waypoint_collision, waypoint_collision_message = scene.check_chain_collision(waypoint_frames)
            if waypoint_collision:
                return {
                    "status": "COLLISION_DETECTED",
                    "robot_type": arm.robot_type,
                    "waypoint": waypoint_label,
                    "details": waypoint_collision_message,
                }

            if waypoint_label.endswith("_END"):
                for sample_index, sample_xyz in enumerate(weld_path_samples):
                    sample_ik_success, sample_q, sample_position_error = arm.inverse_kinematics(
                        target_pos=sample_xyz,
                        target_rot=tool_orientation,
                        q_init=q_seed,
                    )
                    if not sample_ik_success:
                        return {
                            "status": "IK_UNREACHABLE",
                            "robot_type": arm.robot_type,
                            "waypoint": f"{seam_id}_WELD_PATH_SAMPLE_{sample_index}",
                            "target_xyz": sample_xyz.tolist(),
                            "position_error_mm": round(sample_position_error, 3),
                        }

                    sample_frames = arm.forward_kinematics_chain(sample_q)
                    sample_collision, sample_collision_message = scene.check_chain_collision(sample_frames)
                    if sample_collision:
                        return {
                            "status": "COLLISION_DETECTED",
                            "robot_type": arm.robot_type,
                            "waypoint": f"{seam_id}_WELD_PATH_SAMPLE_{sample_index}",
                            "details": sample_collision_message,
                        }

            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=waypoint_label,
                    move_type=move_type,
                    xyz_mm=target_xyz,
                    joints_deg=np.rad2deg(q_solution),
                    arc_state=arc_command,
                    speed_mm_s=waypoint_speed,
                )
            )
            q_seed = q_solution.copy()

    return_home_ik_success, q_return_home, _ = arm.inverse_kinematics(
        target_pos=home_xyz,
        target_rot=home_tool_orientation,
        q_init=q_seed,
    )
    if not return_home_ik_success:
        return {"status": "ERROR", "reason": "Failed to solve IK for RETURN_HOME pose."}

    planned_waypoints.append(
        TrajectoryWaypoint(
            label="RETURN_HOME",
            move_type="MOVJ",
            xyz_mm=home_xyz,
            joints_deg=np.rad2deg(q_return_home),
            speed_mm_s=150.0,
        )
    )

    last_validated_plan = planned_waypoints
    last_robot_type = arm.robot_type

    return {
        "status": "VALIDATED_OK",
        "robot_type": arm.robot_type,
        "dof": arm.dof,
        "seams_planned": seam_ids,
        "total_waypoints": len(planned_waypoints),
        "collision_free": True,
        "weld_path_samples_per_seam": WELD_PATH_COLLISION_SAMPLES,
        "waypoints_summary": [
            {
                "label": wp.label,
                "move": wp.move_type,
                "xyz": np.round(wp.xyz_mm, 1).tolist(),
                "joints": np.round(wp.joints_deg, 1).tolist(),
                "arc": wp.arc_state,
            }
            for wp in planned_waypoints
        ],
    }


@mcp.tool()
def export_robot_program(
    dialect: str = "melfa",
    program_name: str = "WELD_JOB_01",
) -> dict:
    """Compiles the last collision-free validated trajectory into robot controller code."""
    if not last_validated_plan:
        return {
            "status": "ERROR",
            "reason": "No validated trajectory in buffer. Call plan_and_simulate_weld first.",
        }

    normalized_dialect = dialect.lower()
    if normalized_dialect not in SUPPORTED_PROGRAM_DIALECTS:
        return {
            "status": "ERROR",
            "reason": f"Unsupported dialect '{dialect}'. Supported dialects: {sorted(SUPPORTED_PROGRAM_DIALECTS)}.",
        }
    
    program_code = compile_program(
        last_validated_plan,
        dialect=normalized_dialect,
        program_name=program_name,
        robot_type=last_robot_type,
    )

    
    return {
        "status": "COMPILED",
        "robot_type": last_robot_type,
        "dialect": normalized_dialect,
        "line_count": len(program_code.splitlines()),
        "program_code": program_code,
    }
    
if __name__ == "main":
    mcp.run()