from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .kinematics import SUPPORTED_ROBOTS, create_manipulator
from .scene import WorkpieceScene
from .postprocessor import (
    PostprocessorConfig,
    TrajectoryWaypoint,
    compile_program,
)


WELDING_POSTPROCESSOR_CONFIG = PostprocessorConfig(
    override_percent=100.0,
    melfa_arc_start_cmd="ArcStart 1",
    melfa_arc_end_cmd="ArcEnd 1",
    melfa_arc_start_delay_s=0.0,
    melfa_arc_end_delay_s=0.0,
    gcode_arc_start_cmd="M62 P1",
    gcode_arc_end_cmd="M64 P1",
)

HOME_JOINT_SPEED_PERCENT = 40.0
APPROACH_JOINT_SPEED_PERCENT = 40.0

logger = logging.getLogger("mcp_robot_welding.mcp_server")


mcp = MCPServer("Industrial-Robot-Welding-Simulator")

scene = WorkpieceScene()

SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}

MAX_JOINT_JUMP_DEGREES = 15.0
MAX_MOVJ_JOINT_JUMP_DEGREES = 120.0
MAX_MOVL_WAYPOINT_JUMP_DEGREES = 45.0
MAX_WELD_SAMPLE_JOINT_JUMP_DEGREES = 25.0
WELD_PATH_COLLISION_SAMPLES = 25
MOVJ_COLLISION_SAMPLES = 25
MAX_WELD_SPEED_MM_S = 100.0
MAX_APPROACH_CLEARANCE_MM = 250.0
MAX_STORED_PLANS = 20


@dataclass(frozen=True)
class ValidatedPlan:
    plan_id: str
    robot_type: str
    waypoints: list[TrajectoryWaypoint]


validated_plans: dict[str, ValidatedPlan] = {}


def _store_plan(plan: ValidatedPlan) -> None:
    validated_plans[plan.plan_id] = plan

    while len(validated_plans) > MAX_STORED_PLANS:
        oldest_plan_id = next(iter(validated_plans))
        validated_plans.pop(oldest_plan_id, None)


def _get_plan(plan_id: str) -> ValidatedPlan:
    plan = validated_plans.get(plan_id)

    if plan is None:
        raise ToolError(
            f"Unknown or expired plan_id '{plan_id}'. "
            "Call plan_and_simulate_weld first."
        )

    return plan


@mcp.resource("scene://workpiece/seams")
def get_workpiece_seams_resource() -> str:
    """Provides CAD workpiece metadata and available weld seams as JSON."""
    return json.dumps(
        scene.get_workpiece_metadata(),
        indent=2,
    )


@mcp.tool()
def list_supported_robots() -> dict:
    """Lists available manipulator kinematic models and their degrees of freedom."""
    models = []

    for robot_type, description in SUPPORTED_ROBOTS.items():
        manipulator = create_manipulator(robot_type)

        models.append(
            {
                "robot_type": robot_type,
                "dof": manipulator.dof,
                "max_reach_mm": manipulator.max_reach_mm,
                "tool_z_points_into_workpiece": (
                    manipulator.tool_z_points_into_workpiece
                ),
                "description": description,
            }
        )

    return {"supported_robots": models}


@mcp.tool()
def inspect_workpiece() -> dict:
    """Returns workpiece metadata, obstacles, and available weld seam IDs."""
    metadata = scene.get_workpiece_metadata()

    return {
        "part_name": metadata["part_name"],
        "available_seams": [
            {
                "seam_id": seam["seam_id"],
                "description": seam["description"],
                "start_xyz": seam["start_xyz"],
                "end_xyz": seam["end_xyz"],
                "normal_xyz": seam["normal_xyz"],
                "weld_type": seam["weld_type"],
                "length_mm": (
                    scene.seams[seam["seam_id"]].length_mm
                ),
            }
            for seam in metadata["seams"]
        ],
    }


def _validate_planning_arguments(
    seam_ids: list[str],
    weld_speed_mm_s: float,
    approach_clearance_mm: float,
) -> None:
    if not seam_ids:
        raise ToolError("At least one seam_id is required.")

    if len(seam_ids) != len(set(seam_ids)):
        raise ToolError("seam_ids must not contain duplicates.")

    if not np.isfinite(weld_speed_mm_s) or weld_speed_mm_s <= 0.0:
        raise ToolError("weld_speed_mm_s must be a positive finite number.")

    if weld_speed_mm_s > MAX_WELD_SPEED_MM_S:
        raise ToolError(
            f"weld_speed_mm_s must not exceed {MAX_WELD_SPEED_MM_S:.1f} mm/s."
        )

    if not np.isfinite(approach_clearance_mm) or approach_clearance_mm < 0.0:
        raise ToolError(
            "approach_clearance_mm must be a non-negative finite number."
        )

    if approach_clearance_mm > MAX_APPROACH_CLEARANCE_MM:
        raise ToolError(
            "approach_clearance_mm must not exceed "
            f"{MAX_APPROACH_CLEARANCE_MM:.1f} mm."
        )


def _validate_seam_ids(seam_ids: list[str]) -> None:
    unknown_seams = [
        seam_id for seam_id in seam_ids if seam_id not in scene.seams
    ]

    if unknown_seams:
        raise ToolError(
            f"Unknown seam_id(s): {unknown_seams}. Available seam IDs: "
            f"{list(scene.seams.keys())}."
        )


def _ik_failure_response(
    robot_type: str,
    waypoint: str,
    target_xyz: np.ndarray,
    position_error: float,
) -> dict:
    return {
        "status": "IK_UNREACHABLE",
        "robot_type": robot_type,
        "waypoint": waypoint,
        "target_xyz": np.round(target_xyz, 3).tolist(),
        "position_error_mm": round(float(position_error), 3),
        "details": (
            "IK did not reach the requested position/orientation tolerance. "
            "The target may be outside the reachable workspace, blocked by "
            "joint limits, or require a different tool orientation."
        ),
    }


def _collision_failure_response(
    robot_type: str,
    waypoint: str,
    collision_message: str,
    obstacle_name: str | None,
    link_name: str | None,
) -> dict:
    return {
        "status": "COLLISION_DETECTED",
        "robot_type": robot_type,
        "waypoint": waypoint,
        "details": collision_message,
        "obstacle": obstacle_name,
        "link": link_name,
    }


@mcp.tool()
def plan_and_simulate_weld(
    seam_ids: list[str],
    weld_speed_mm_s: float = 25.0,
    approach_clearance_mm: float = 40.0,
    robot_type: str = "industrial_6dof",
) -> dict:
    """
    Plans approach, weld, retract, and return-home trajectories for seams.

    The result is a kinematically and statically collision-checked simulation.
    It is not a production-ready robot program until the robot mounting, tool
    geometry, controller postprocessor, and welding-interface mapping have
    been validated.
    """
    _validate_planning_arguments(
        seam_ids,
        weld_speed_mm_s,
        approach_clearance_mm,
    )
    _validate_seam_ids(seam_ids)

    try:
        arm = create_manipulator(robot_type)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    planned_waypoints: list[TrajectoryWaypoint] = []
    home_xyz = np.array([450.0, 0.0, 420.0], dtype=np.float64)
    home_tool_orientation = arm.default_tool_orientation()
    safe_transit_z = 430.0 if arm.position_only_orientation else 650.0
    safe_transit_xyz = np.array([450.0, 0.0, safe_transit_z], dtype=np.float64)


    def _append_safe_transit(
        q_prev: np.ndarray,
        transit_label: str,
    ) -> tuple[np.ndarray, list[np.ndarray]] | None:
        ik_ok, q_transit, _ = arm.inverse_kinematics(
            target_pos=safe_transit_xyz,
            target_rot=home_tool_orientation,
            q_init=q_prev,
            max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
        )
        if not ik_ok:
            return None

        transit_frames = arm.forward_kinematics_chain(q_transit)
        if scene.check_chain_collision(transit_frames, allow_tool_contact=False).collided:
            return None

        if _check_movj_collision(
            arm=arm,
            q_start=q_prev,
            q_end=q_transit,
            waypoint_label=transit_label,
            robot_type=arm.robot_type,
        ) is not None:
            return None

        planned_waypoints.append(
            TrajectoryWaypoint(
                label=transit_label,
                move_type="MOVJ",
                xyz_mm=safe_transit_xyz,
                joints_deg=np.rad2deg(q_transit),
                orientation=home_tool_orientation,
                joint_speed_percent=HOME_JOINT_SPEED_PERCENT,
            )
        )
        return q_transit.copy(), transit_frames


    home_ik_success, q_home, home_position_error = arm.inverse_kinematics(
        target_pos=home_xyz,
        target_rot=home_tool_orientation,
    )

    if not home_ik_success:
        return _ik_failure_response(
            robot_type=arm.robot_type,
            waypoint="HOME_SAFE",
            target_xyz=home_xyz,
            position_error=home_position_error,
        )

    planned_waypoints.append(
        TrajectoryWaypoint(
            label="HOME_SAFE",
            move_type="MOVJ",
            xyz_mm=home_xyz,
            joints_deg=np.rad2deg(q_home),
            orientation=home_tool_orientation,
            joint_speed_percent=APPROACH_JOINT_SPEED_PERCENT,
        )
    )

    q_previous = q_home.copy()
    home_frames = arm.forward_kinematics_chain(q_home)
    previous_frames = home_frames
    
    for seam_idx, seam_id in enumerate(seam_ids):
        if seam_idx > 0:
            transit_err = _check_movj_collision(
                arm=arm,
                q_start=q_previous,
                q_end=q_home,
                waypoint_label=f"{seam_id}_SAFE_TRANSIT",
                robot_type=arm.robot_type,
            )
            if transit_err is not None:
                return transit_err
            
            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_SAFE_TRANSIT",
                    move_type="MOVJ",
                    xyz_mm=home_xyz,
                    joints_deg=np.rad2deg(q_home),
                    orientation=home_tool_orientation,
                    joint_speed_percent=HOME_JOINT_SPEED_PERCENT,
                )
            )
            previous_frames = home_frames
            q_previous = q_home.copy()

        seam = scene.seams[seam_id]
        seam_start = seam.start
        seam_end = seam.end
        seam_normal = seam.normal

        approach_xyz = seam_start + seam_normal * approach_clearance_mm
        retract_xyz = seam_end + seam_normal * approach_clearance_mm

        if arm.tool_z_points_into_workpiece:
            tool_z_direction = -seam_normal
        else:
            tool_z_direction = seam_normal

        current_tool_x = previous_frames[-1][:3, 0]
        tool_orientation = arm.make_tool_orientation(
            tool_z_direction,
            reference_x_axis=current_tool_x,
        )
        weld_path_samples = seam.path_points(WELD_PATH_COLLISION_SAMPLES)

        weld_steps = [
            (
                f"{seam_id}_APPROACH",
                "MOVJ",
                approach_xyz,
                None,
                120.0,
            ),
            (
                f"{seam_id}_START",
                "MOVL",
                seam_start,
                "ARC_ON",
                weld_speed_mm_s,
            ),
            (
                f"{seam_id}_MID",
                "MOVL",
                0.5 * (seam_start + seam_end),
                None,
                weld_speed_mm_s,
            ),
            (
                f"{seam_id}_END",
                "MOVL",
                seam_end,
                "ARC_OFF",
                weld_speed_mm_s,
            ),
            (
                f"{seam_id}_RETRACT",
                "MOVJ",
                retract_xyz,
                None,
                120.0,
            ),
        ]

        for waypoint_label, move_type, target_xyz, arc_command, speed in weld_steps:
            allowed_jump_deg = (
                MAX_MOVJ_JOINT_JUMP_DEGREES
                if move_type == "MOVJ"
                else MAX_MOVL_WAYPOINT_JUMP_DEGREES
            )

            
            ik_success, q_solution, position_error = arm.inverse_kinematics(
                target_pos=target_xyz,
                target_rot=tool_orientation,
                q_init=q_previous,
                max_joint_jump_deg=allowed_jump_deg,
            )

            if not ik_success:
                return _ik_failure_response(
                    robot_type=arm.robot_type,
                    waypoint=waypoint_label,
                    target_xyz=target_xyz,
                    position_error=position_error,
                )

            joint_jump_deg = arm.joint_jump(q_solution, q_previous)

            if joint_jump_deg > allowed_jump_deg:
                return {
                    "status": "JOINT_DISCONTINUITY",
                    "robot_type": arm.robot_type,
                    "waypoint": waypoint_label,
                    "joint_jump_deg": round(joint_jump_deg, 3),
                    "maximum_allowed_deg": allowed_jump_deg,
                    "details": (
                        "The selected IK branch creates a large joint-space "
                        "jump from the previous waypoint."
                    ),
                }

            waypoint_frames = arm.forward_kinematics_chain(q_solution)
            collision_result = scene.check_chain_collision(
                waypoint_frames,
                allow_tool_contact=(
                    waypoint_label.endswith("_START")
                    or waypoint_label.endswith("_MID")
                    or waypoint_label.endswith("_END")
                ),
                tool_contact_clearance_mm=0.0,
            )


            if collision_result.collided:
                return _collision_failure_response(
                    robot_type=arm.robot_type,
                    waypoint=waypoint_label,
                    collision_message=collision_result.message,
                    obstacle_name=collision_result.obstacle_name,
                    link_name=collision_result.link_name,
                )

            if move_type == "MOVJ":
                traj_err = _check_movj_collision(
                    arm=arm,
                    q_start=q_previous,
                    q_end=q_solution,
                    waypoint_label=waypoint_label,
                    robot_type=arm.robot_type,
                    allow_tool_contact=waypoint_label.endswith("_RETRACT"),
                )
                if traj_err is not None:
                    return traj_err




            if waypoint_label.endswith("_END"):
                q_weld_sample = q_previous.copy()

                for sample_index, sample_xyz in enumerate(weld_path_samples):
                    sample_ik_success, sample_q, sample_error = (
                        arm.inverse_kinematics(
                            target_pos=sample_xyz,
                            target_rot=tool_orientation,
                            q_init=q_weld_sample,
                            max_iters=800,
                            pos_tol=0.2,
                            max_joint_jump_deg=MAX_WELD_SAMPLE_JOINT_JUMP_DEGREES,
                        )
                    )


                    if not sample_ik_success:
                        return _ik_failure_response(
                            robot_type=arm.robot_type,
                            waypoint=(
                                f"{seam_id}_WELD_PATH_SAMPLE_{sample_index}"
                            ),
                            target_xyz=sample_xyz,
                            position_error=sample_error,
                        )

                    sample_jump = arm.joint_jump(sample_q, q_weld_sample)

                    if sample_jump > MAX_WELD_SAMPLE_JOINT_JUMP_DEGREES:
                        return {
                            "status": "JOINT_DISCONTINUITY",
                            "robot_type": arm.robot_type,
                            "waypoint": (
                                f"{seam_id}_WELD_PATH_SAMPLE_{sample_index}"
                            ),
                            "joint_jump_deg": round(sample_jump, 3),
                            "maximum_allowed_deg": MAX_WELD_SAMPLE_JOINT_JUMP_DEGREES,
                            "details": (
                                "Adjacent weld-path IK solutions select "
                                "different branches."
                            ),
                        }

                    sample_frames = arm.forward_kinematics_chain(sample_q)
                    sample_collision = scene.check_chain_collision(
                        sample_frames,
                        allow_tool_contact=True,
                        tool_contact_clearance_mm=0.0,
                    )

                    if sample_collision.collided:
                        return _collision_failure_response(
                            robot_type=arm.robot_type,
                            waypoint=(
                                f"{seam_id}_WELD_PATH_SAMPLE_{sample_index}"
                            ),
                            collision_message=sample_collision.message,
                            obstacle_name=sample_collision.obstacle_name,
                            link_name=sample_collision.link_name,
                        )

                    q_weld_sample = sample_q.copy()

            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=waypoint_label,
                    move_type=move_type,
                    xyz_mm=target_xyz,
                    joints_deg=np.rad2deg(q_solution),
                    orientation=tool_orientation,
                    arc_state=arc_command,
                    tcp_speed_mm_s=(
                        speed if move_type == "MOVL" else None
                    ),
                    joint_speed_percent=(
                        HOME_JOINT_SPEED_PERCENT if move_type == "MOVJ" else None
                    ),
                )
            )
            
            previous_frames = waypoint_frames
            q_previous = q_solution.copy()
        
        transit_res = _append_safe_transit(q_previous, f"{seam_id}_SAFE_TRANSIT")
        if transit_res is None:
            return {
                "status": "SAFE_TRANSIT_UNAVAILABLE",
                "robot_type": arm.robot_type,
                "waypoint": f"{seam_id}_SAFE_TRANSIT",
                "details": (
                    "Could not find a collision-free transit configuration "
                    "above the workpiece."
                ),
            }
        q_previous, previous_frames = transit_res
    
    # <--- ВСТАВИТЬ СЮДА (перед RETURN_HOME):
    final_transit_result = _append_safe_transit(
        q_previous,
        "FINAL_SAFE_TRANSIT",
    )

    if final_transit_result is None:
        return {
            "status": "SAFE_TRANSIT_UNAVAILABLE",
            "robot_type": arm.robot_type,
            "waypoint": "FINAL_SAFE_TRANSIT",
            "details": (
                "Could not find a collision-free transit configuration "
                "before returning home."
            ),
        }

    q_previous, previous_frames = final_transit_result
    
    return_home_ik_success, q_return_home, return_position_error = (
        arm.inverse_kinematics(
            target_pos=home_xyz,
            target_rot=home_tool_orientation,
            q_init=q_previous,
            max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
        )
    )

    if not return_home_ik_success:
        return _ik_failure_response(
            robot_type=arm.robot_type,
            waypoint="RETURN_HOME",
            target_xyz=home_xyz,
            position_error=return_position_error,
        )

    return_traj_err = _check_movj_collision(
        arm=arm,
        q_start=q_previous,
        q_end=q_return_home,
        waypoint_label="RETURN_HOME",
        robot_type=arm.robot_type,
    )
    if return_traj_err is not None:
        return return_traj_err
    
    planned_waypoints.append(
        TrajectoryWaypoint(
            label="RETURN_HOME",
            move_type="MOVJ",
            xyz_mm=home_xyz,
            joints_deg=np.rad2deg(q_return_home),
            orientation=home_tool_orientation,
            joint_speed_percent=APPROACH_JOINT_SPEED_PERCENT,
        )
    )

    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    plan = ValidatedPlan(
        plan_id=plan_id,
        robot_type=arm.robot_type,
        waypoints=planned_waypoints,
    )

    _store_plan(plan)

    return {
        "status": "VALIDATED_OK",
        "validation_level": "kinematic_and_static_collision",
        "plan_id": plan_id,
        "robot_type": arm.robot_type,
        "dof": arm.dof,
        "seams_planned": seam_ids,
        "total_waypoints": len(planned_waypoints),
        "weld_path_samples_per_seam": WELD_PATH_COLLISION_SAMPLES,
        "warnings": [
            "Collision checking uses conservative capsule approximations, "
            "not mesh-accurate robot or tool geometry.",
            "Validate the robot mounting transform, tool frame, controller "
            "postprocessor, and welding-interface commands before production."
        ],
        "waypoints_summary": [
            {
                "label": waypoint.label,
                "move": waypoint.move_type,
                "xyz": np.round(waypoint.xyz_mm, 1).tolist(),
                "joints": np.round(waypoint.joints_deg, 1).tolist(),
                "arc": waypoint.arc_state,
                "tcp_speed_mm_s": waypoint.tcp_speed_mm_s,
                "joint_speed_percent": waypoint.joint_speed_percent,
            }
            for waypoint in planned_waypoints
        ],

    }

def _check_movj_collision(
    arm,
    q_start: np.ndarray,
    q_end: np.ndarray,
    waypoint_label: str,
    robot_type: str,
    allow_tool_contact: bool = False,
) -> dict | None:
    for alpha in np.linspace(0.0, 1.0, MOVJ_COLLISION_SAMPLES):
        q_interp = (1.0 - alpha) * q_start + alpha * q_end
        interp_frames = arm.forward_kinematics_chain(q_interp)
        res = scene.check_chain_collision(
            interp_frames,
            allow_tool_contact=allow_tool_contact,
        )
        if res.collided:
            return _collision_failure_response(
                robot_type=robot_type,
                waypoint=f"{waypoint_label}_TRAJECTORY",
                collision_message=f"{res.message} (alpha={alpha:.3f})",
                obstacle_name=res.obstacle_name,
                link_name=res.link_name,
            )
    return None


@mcp.tool()
def export_robot_program(
    plan_id: str,
    dialect: str = "melfa",
    program_name: str = "WELD_JOB_01",
) -> dict:
    """Compiles a previously validated trajectory into robot-controller code."""
    plan = _get_plan(plan_id)

    normalized_dialect = dialect.strip().lower()

    if normalized_dialect not in SUPPORTED_PROGRAM_DIALECTS:
        raise ToolError(
            f"Unsupported dialect '{dialect}'. Supported dialects: "
            f"{sorted(SUPPORTED_PROGRAM_DIALECTS)}."
        )

    try:
        program_code = compile_program(
            plan.waypoints,
            dialect=normalized_dialect,
            program_name=program_name,
            robot_type=plan.robot_type,
            config=WELDING_POSTPROCESSOR_CONFIG,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    return {
        "status": "COMPILED",
        "plan_id": plan.plan_id,
        "robot_type": plan.robot_type,
        "dialect": normalized_dialect,
        "line_count": len(program_code.splitlines()),
        "program_code": program_code,
    }


if __name__ == "__main__":
    mcp.run()