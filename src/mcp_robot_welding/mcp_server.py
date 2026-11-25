from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .kinematics import (
    _get_robot_registry,
    _refresh_supported_robots,
    create_manipulator,
)
from .postprocessor import (
    PostprocessorConfig,
    TrajectoryWaypoint,
    compile_program,
)
from .scene import WorkpieceScene

WELDING_POSTPROCESSOR_CONFIG = PostprocessorConfig(
    override_percent=100.0,
    melfa_arc_start_cmd="ArcStart 1",
    melfa_arc_end_cmd="ArcEnd 1",
    melfa_arc_start_delay_s=0.0,
    melfa_arc_end_delay_s=0.0,
    gcode_arc_start_cmd="M62 P1",
    gcode_arc_end_cmd="M64 P1",
)


@asynccontextmanager
async def server_lifespan(server):
    _refresh_supported_robots()
    yield


logger = logging.getLogger(
    "mcp_robot_welding.mcp_server",
)


mcp = MCPServer(
    "Industrial-Robot-Welding-Simulator",
    lifespan=server_lifespan,
)

scene = WorkpieceScene()

SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}

HOME_JOINT_SPEED_PERCENT = 40.0
APPROACH_JOINT_SPEED_PERCENT = 40.0
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


_plans_lock = threading.Lock()


def _store_plan(plan: ValidatedPlan) -> None:
    with _plans_lock:
        validated_plans[plan.plan_id] = plan

        while len(validated_plans) > MAX_STORED_PLANS:
            oldest_plan_id = next(iter(validated_plans))
            validated_plans.pop(oldest_plan_id, None)


def _get_plan(plan_id: str) -> ValidatedPlan:
    with _plans_lock:
        plan = validated_plans.get(plan_id)

    if plan is None:
        raise ToolError(f"Unknown or expired plan_id '{plan_id}'.")

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
    registry = _get_robot_registry()

    models = []

    for robot_type, config in registry.items():
        models.append(
            {
                "robot_type": robot_type,
                "description": str(config.get("description", robot_type)),
                "dof": len(config["dh_table"]),
                "max_reach_mm": float(config["max_reach_mm"]),
                "position_only_orientation": bool(
                    config.get(
                        "position_only_orientation",
                        False,
                    )
                ),
                "tool_z_points_into_workpiece": bool(
                    config.get(
                        "tool_z_points_into_workpiece",
                        True,
                    )
                ),
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
                "length_mm": (scene.seams[seam["seam_id"]].length_mm),
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
        raise ToolError("approach_clearance_mm must be a non-negative finite number.")

    if approach_clearance_mm > MAX_APPROACH_CLEARANCE_MM:
        raise ToolError(
            "approach_clearance_mm must not exceed "
            f"{MAX_APPROACH_CLEARANCE_MM:.1f} mm."
        )


def _validate_seam_ids(seam_ids: list[str]) -> None:
    unknown_seams = [seam_id for seam_id in seam_ids if seam_id not in scene.seams]

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
    orientation_error_rad: float | None = None,
) -> dict:
    response = {
        "status": "IK_UNREACHABLE",
        "robot_type": robot_type,
        "waypoint": waypoint,
        "target_xyz": np.round(target_xyz, 3).tolist(),
        "position_error_mm": round(float(position_error), 3),
        "details": ("IK did not reach the requested position/orientation tolerance."),
    }

    if orientation_error_rad is not None:
        response["orientation_error_rad"] = round(
            float(orientation_error_rad),
            6,
        )
        response["orientation_error_deg"] = round(
            float(np.rad2deg(orientation_error_rad)),
            3,
        )

    return response


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
    Plans approach, weld, retract, transit, and return-home trajectories
    for requested weld seams.

    The result is a kinematically and statically collision-checked simulation.
    It is not production-ready until the robot mounting, tool geometry,
    controller postprocessor, and welding-interface mapping are validated.
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

    home_xyz = np.array(
        [450.0, 0.0, 420.0],
        dtype=np.float64,
    )
    home_tool_orientation = arm.default_tool_orientation()

    safe_transit_xyz = scene.get_safe_transit_position(robot_type=arm.robot_type)

    def append_waypoint(
        label: str,
        move_type: str,
        target_xyz: np.ndarray,
        joints_rad: np.ndarray,
        orientation: np.ndarray,
        *,
        arc_state: str | None = None,
        tcp_speed_mm_s: float | None = None,
        joint_speed_percent: float | None = None,
    ) -> None:
        planned_waypoints.append(
            TrajectoryWaypoint(
                label=label,
                move_type=move_type,
                xyz_mm=target_xyz,
                joints_deg=np.rad2deg(joints_rad),
                orientation=orientation,
                arc_state=arc_state,
                tcp_speed_mm_s=tcp_speed_mm_s,
                joint_speed_percent=joint_speed_percent,
            )
        )

    def check_and_append_movj(
        label: str,
        target_xyz: np.ndarray,
        q_start: np.ndarray,
        q_end: np.ndarray,
        orientation: np.ndarray,
        *,
        allow_tool_contact: bool = False,
    ) -> dict[str, Any] | None:
        end_frames = arm.forward_kinematics_chain(q_end)

        collision_result = scene.check_chain_collision(
            end_frames,
            allow_tool_contact=allow_tool_contact,
            tool_contact_clearance_mm=0.0,
        )

        if collision_result.collided:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=label,
                collision_message=collision_result.message,
                obstacle_name=collision_result.obstacle_name,
                link_name=collision_result.link_name,
            )

        trajectory_result = scene.check_trajectory_collision(
            arm.forward_kinematics_chain(q_start),
            end_frames,
            samples=MOVJ_COLLISION_SAMPLES,
            allow_tool_contact=allow_tool_contact,
            tool_contact_clearance_mm=0.0,
        )

        if trajectory_result.collided:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{label}_TRAJECTORY",
                collision_message=trajectory_result.message,
                obstacle_name=trajectory_result.obstacle_name,
                link_name=trajectory_result.link_name,
            )

        append_waypoint(
            label=label,
            move_type="MOVJ",
            target_xyz=target_xyz,
            joints_rad=q_end,
            orientation=orientation,
            joint_speed_percent=APPROACH_JOINT_SPEED_PERCENT,
        )

        return None

    def append_safe_transit(
        q_previous: np.ndarray,
        previous_frames: list[np.ndarray],
        transit_label: str,
    ) -> tuple[
        np.ndarray | None,
        list[np.ndarray] | None,
        dict[str, Any] | None,
    ]:
        ik_success, q_transit, position_error = arm.inverse_kinematics(
            target_pos=safe_transit_xyz,
            target_rot=home_tool_orientation,
            q_init=q_previous,
            max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
        )

        if not ik_success:
            return (
                None,
                None,
                _ik_failure_response(
                    robot_type=arm.robot_type,
                    waypoint=transit_label,
                    target_xyz=safe_transit_xyz,
                    position_error=position_error,
                ),
            )

        transit_frames = arm.forward_kinematics_chain(q_transit)

        collision_result = scene.check_chain_collision(
            transit_frames,
            allow_tool_contact=False,
            tool_contact_clearance_mm=0.0,
        )

        if collision_result.collided:
            return (
                None,
                None,
                _collision_failure_response(
                    robot_type=arm.robot_type,
                    waypoint=transit_label,
                    collision_message=collision_result.message,
                    obstacle_name=collision_result.obstacle_name,
                    link_name=collision_result.link_name,
                ),
            )

        trajectory_result = scene.check_trajectory_collision(
            previous_frames,
            transit_frames,
            samples=MOVJ_COLLISION_SAMPLES,
            allow_tool_contact=False,
            tool_contact_clearance_mm=0.0,
        )

        if trajectory_result.collided:
            return (
                None,
                None,
                _collision_failure_response(
                    robot_type=arm.robot_type,
                    waypoint=f"{transit_label}_TRAJECTORY",
                    collision_message=trajectory_result.message,
                    obstacle_name=trajectory_result.obstacle_name,
                    link_name=trajectory_result.link_name,
                ),
            )

        append_waypoint(
            label=transit_label,
            move_type="MOVJ",
            target_xyz=safe_transit_xyz,
            joints_rad=q_transit,
            orientation=home_tool_orientation,
            joint_speed_percent=HOME_JOINT_SPEED_PERCENT,
        )

        return (
            q_transit.copy(),
            transit_frames,
            None,
        )

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

    home_frames = arm.forward_kinematics_chain(q_home)

    home_collision_result = scene.check_chain_collision(
        home_frames,
        allow_tool_contact=False,
        tool_contact_clearance_mm=0.0,
    )

    if home_collision_result.collided:
        return _collision_failure_response(
            robot_type=arm.robot_type,
            waypoint="HOME_SAFE",
            collision_message=home_collision_result.message,
            obstacle_name=home_collision_result.obstacle_name,
            link_name=home_collision_result.link_name,
        )

    append_waypoint(
        label="HOME_SAFE",
        move_type="MOVJ",
        target_xyz=home_xyz,
        joints_rad=q_home,
        orientation=home_tool_orientation,
        joint_speed_percent=APPROACH_JOINT_SPEED_PERCENT,
    )

    q_previous = q_home.copy()
    previous_frames = home_frames

    for seam_id in seam_ids:
        seam = scene.seams[seam_id]
        seam_start = seam.start
        seam_end = seam.end
        seam_normal = seam.normal

        seam_length = float(np.linalg.norm(seam_end - seam_start))

        if seam_length < 1e-9:
            return {
                "status": "ERROR",
                "reason": (f"Seam '{seam_id}' has zero length."),
            }

        # seam_tangent = (seam_end - seam_start) / seam_length

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

        # APPROACH
        approach_ik_success, q_approach, approach_error = arm.inverse_kinematics(
            target_pos=approach_xyz,
            target_rot=tool_orientation,
            q_init=q_previous,
            max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
        )

        if not approach_ik_success:
            return _ik_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_APPROACH",
                target_xyz=approach_xyz,
                position_error=approach_error,
            )

        approach_error_response = check_and_append_movj(
            label=f"{seam_id}_APPROACH",
            target_xyz=approach_xyz,
            q_start=q_previous,
            q_end=q_approach,
            orientation=tool_orientation,
            allow_tool_contact=False,
        )

        if approach_error_response is not None:
            return approach_error_response

        q_previous = q_approach.copy()
        previous_frames = arm.forward_kinematics_chain(q_approach)

        # WELD START
        start_ik_success, q_start, start_error = arm.inverse_kinematics(
            target_pos=seam_start,
            target_rot=tool_orientation,
            q_init=q_previous,
            max_iters=1500,
            pos_tol=0.05,
            rot_tol=0.01,
            max_joint_jump_deg=MAX_MOVL_WAYPOINT_JUMP_DEGREES,
        )

        if not start_ik_success:
            return _ik_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_START",
                target_xyz=seam_start,
                position_error=start_error,
            )

        start_frames = arm.forward_kinematics_chain(q_start)

        start_collision_result = scene.check_chain_collision(
            start_frames,
            allow_tool_contact=True,
            tool_contact_clearance_mm=0.0,
        )

        if start_collision_result.collided:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_START",
                collision_message=start_collision_result.message,
                obstacle_name=start_collision_result.obstacle_name,
                link_name=start_collision_result.link_name,
            )

        trajectory_result, q_weld_end = scene.check_cartesian_trajectory_collision(
            arm=arm,
            q_start=q_start,
            target_positions=weld_path_samples,
            target_orientation=tool_orientation,
            samples=WELD_PATH_COLLISION_SAMPLES,
            allow_tool_contact=True,
            tool_contact_clearance_mm=0.0,
            max_joint_jump_deg=(MAX_WELD_SAMPLE_JOINT_JUMP_DEGREES),
            start_tolerance_mm=0.2,
        )

        if trajectory_result.collided or q_weld_end is None:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_WELD_TRAJECTORY",
                collision_message=trajectory_result.message,
                obstacle_name=trajectory_result.obstacle_name,
                link_name=trajectory_result.link_name,
            )

        append_waypoint(
            label=f"{seam_id}_START",
            move_type="MOVL",
            target_xyz=seam_start,
            joints_rad=q_start,
            orientation=tool_orientation,
            arc_state="ARC_ON",
            tcp_speed_mm_s=weld_speed_mm_s,
        )

        q_previous = q_start.copy()
        previous_frames = start_frames

        # WELD MIDPOINT
        midpoint_xyz = 0.5 * (seam_start + seam_end)

        midpoint_ik_success, q_midpoint, midpoint_error = arm.inverse_kinematics(
            target_pos=midpoint_xyz,
            target_rot=tool_orientation,
            q_init=q_previous,
            max_iters=800,
            pos_tol=0.2,
            max_joint_jump_deg=MAX_MOVL_WAYPOINT_JUMP_DEGREES,
        )

        if not midpoint_ik_success:
            return _ik_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_MID",
                target_xyz=midpoint_xyz,
                position_error=midpoint_error,
            )

        midpoint_frames = arm.forward_kinematics_chain(q_midpoint)

        midpoint_collision_result = scene.check_chain_collision(
            midpoint_frames,
            allow_tool_contact=True,
            tool_contact_clearance_mm=0.0,
        )

        if midpoint_collision_result.collided:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_MID",
                collision_message=midpoint_collision_result.message,
                obstacle_name=midpoint_collision_result.obstacle_name,
                link_name=midpoint_collision_result.link_name,
            )

        append_waypoint(
            label=f"{seam_id}_MID",
            move_type="MOVL",
            target_xyz=midpoint_xyz,
            joints_rad=q_midpoint,
            orientation=tool_orientation,
            tcp_speed_mm_s=weld_speed_mm_s,
        )

        q_previous = q_midpoint.copy()
        previous_frames = midpoint_frames

        # WELD END
        end_ik_success, q_end, end_error = arm.inverse_kinematics(
            target_pos=seam_end,
            target_rot=tool_orientation,
            q_init=q_weld_end,
            max_iters=800,
            pos_tol=0.2,
            rot_tol=0.02,
            max_joint_jump_deg=MAX_MOVL_WAYPOINT_JUMP_DEGREES,
        )

        if not end_ik_success:
            return _ik_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_END",
                target_xyz=seam_end,
                position_error=end_error,
            )

        end_frames = arm.forward_kinematics_chain(q_end)

        end_collision_result = scene.check_chain_collision(
            end_frames,
            allow_tool_contact=True,
            tool_contact_clearance_mm=0.0,
        )

        if end_collision_result.collided:
            return _collision_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_END",
                collision_message=end_collision_result.message,
                obstacle_name=end_collision_result.obstacle_name,
                link_name=end_collision_result.link_name,
            )

        append_waypoint(
            label=f"{seam_id}_END",
            move_type="MOVL",
            target_xyz=seam_end,
            joints_rad=q_end,
            orientation=tool_orientation,
            arc_state="ARC_OFF",
            tcp_speed_mm_s=weld_speed_mm_s,
        )

        q_previous = q_end.copy()
        previous_frames = end_frames

        # RETRACT
        retract_ik_success, q_retract, retract_error = arm.inverse_kinematics(
            target_pos=retract_xyz,
            target_rot=tool_orientation,
            q_init=q_previous,
            max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
        )

        if not retract_ik_success:
            return _ik_failure_response(
                robot_type=arm.robot_type,
                waypoint=f"{seam_id}_RETRACT",
                target_xyz=retract_xyz,
                position_error=retract_error,
            )

        retract_error_response = check_and_append_movj(
            label=f"{seam_id}_RETRACT",
            target_xyz=retract_xyz,
            q_start=q_previous,
            q_end=q_retract,
            orientation=tool_orientation,
            allow_tool_contact=True,
        )

        if retract_error_response is not None:
            return retract_error_response

        q_previous = q_retract.copy()
        previous_frames = arm.forward_kinematics_chain(q_retract)

        # SAFE TRANSIT AFTER SEAM
        (
            q_transit,
            transit_frames,
            transit_error_response,
        ) = append_safe_transit(
            q_previous=q_previous,
            previous_frames=previous_frames,
            transit_label=f"{seam_id}_SAFE_TRANSIT",
        )

        if transit_error_response is not None:
            return transit_error_response

        q_previous = q_transit
        previous_frames = transit_frames

    # FINAL SAFE TRANSIT BEFORE HOME
    (
        q_transit,
        transit_frames,
        transit_error_response,
    ) = append_safe_transit(
        q_previous=q_previous,
        previous_frames=previous_frames,
        transit_label="FINAL_SAFE_TRANSIT",
    )

    if transit_error_response is not None:
        return transit_error_response

    q_previous = q_transit
    previous_frames = transit_frames

    # RETURN HOME
    return_home_ik_success, q_return_home, return_error = arm.inverse_kinematics(
        target_pos=home_xyz,
        target_rot=home_tool_orientation,
        q_init=q_previous,
        max_joint_jump_deg=MAX_MOVJ_JOINT_JUMP_DEGREES,
    )

    if not return_home_ik_success:
        return _ik_failure_response(
            robot_type=arm.robot_type,
            waypoint="RETURN_HOME",
            target_xyz=home_xyz,
            position_error=return_error,
        )

    return_home_error_response = check_and_append_movj(
        label="RETURN_HOME",
        target_xyz=home_xyz,
        q_start=q_previous,
        q_end=q_return_home,
        orientation=home_tool_orientation,
        allow_tool_contact=False,
    )

    if return_home_error_response is not None:
        return return_home_error_response

    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    plan = ValidatedPlan(
        plan_id=plan_id,
        robot_type=arm.robot_type,
        waypoints=planned_waypoints,
    )

    _store_plan(plan)

    return {
        "status": "VALIDATED_OK",
        "validation_level": ("kinematic_and_static_collision"),
        "plan_id": plan_id,
        "robot_type": arm.robot_type,
        "dof": arm.dof,
        "seams_planned": seam_ids,
        "total_waypoints": len(planned_waypoints),
        "weld_path_samples_per_seam": (WELD_PATH_COLLISION_SAMPLES),
        "warnings": [
            (
                "Collision checking uses conservative capsule "
                "approximations, not mesh-accurate robot or tool "
                "geometry."
            ),
            (
                "Validate the robot mounting transform, tool frame, "
                "controller postprocessor, and welding-interface "
                "commands before production."
            ),
        ],
        "waypoints_summary": [
            {
                "label": waypoint.label,
                "move": waypoint.move_type,
                "xyz": np.round(
                    waypoint.xyz_mm,
                    1,
                ).tolist(),
                "joints": np.round(
                    waypoint.joints_deg,
                    1,
                ).tolist(),
                "arc": waypoint.arc_state,
                "tcp_speed_mm_s": waypoint.tcp_speed_mm_s,
                "joint_speed_percent": (waypoint.joint_speed_percent),
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
