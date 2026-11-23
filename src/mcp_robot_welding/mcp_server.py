from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import uuid
import numpy as np
from mcp.server.mcpserver import MCPServer

from .kinematics import SUPPORTED_ROBOTS, create_manipulator
from .postprocessor import TrajectoryWaypoint, compile_program
from .scene import WorkpieceScene

logger = logging.getLogger("mcp_robot_welding.server")

mcp = MCPServer("Industrial-Robot-Welding-Simulator")

scene = WorkpieceScene()

SUPPORTED_PROGRAM_DIALECTS = {"melfa", "gcode"}
MAX_WAYPOINT_JOINT_JUMP_DEGREES = 120.0
MAX_SAMPLE_JOINT_JUMP_DEGREES = 25.0
WELD_PATH_COLLISION_SAMPLES = 20
MOVJ_COLLISION_SAMPLES = 15
MAX_WELD_SPEED_MM_S = 250.0
MAX_APPROACH_CLEARANCE_MM = 300.0
PROGRAM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class ValidatedPlanRecord:
    plan_id: str
    robot_type: str
    dof: int
    seams_planned: list[str]
    waypoints: tuple[TrajectoryWaypoint, ...]


# Thread/request-safe registry keyed by plan_id (with latest_plan_id fallback for single-agent CLI)
validated_plans: dict[str, ValidatedPlanRecord] = {}
latest_plan_id: str | None = None


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


def _check_joint_space_segment_collisions(
    arm,
    q_start: np.ndarray,
    q_end: np.ndarray,
    segment_label: str,
    robot_type: str,
    samples: int = MOVJ_COLLISION_SAMPLES,
) -> dict | None:
    """Interpolates joint configurations for MOVJ segments and checks every intermediate pose."""
    for idx, alpha in enumerate(np.linspace(0.0, 1.0, samples)):
        q_interp = (1.0 - alpha) * q_start + alpha * q_end
        frames = arm.forward_kinematics_chain(q_interp)
        collision, msg = scene.check_chain_collision(frames)
        if collision:
            return {
                "status": "COLLISION_DETECTED",
                "robot_type": robot_type,
                "waypoint": f"{segment_label}_MOVJ_SAMPLE_{idx}",
                "details": msg,
                "partial_plan_retained": False,
            }
    return None


@mcp.tool()
def plan_and_simulate_weld(
    seam_ids: list[str],
    weld_speed_mm_s: float = 25.0,
    approach_clearance_mm: float = 40.0,
    robot_type: str = "industrial_6dof",
    tool_z_points_into_workpiece: bool = True,
) -> dict:
    """
    Plans approach, linear weld, and retract trajectories for requested seams
    with full continuous MOVJ/MOVL collision checking and branch-continuous IK.

    Convention: when tool_z_points_into_workpiece=True, tool +Z aligns with -seam_normal.
    """
    global latest_plan_id

    try:
        if not seam_ids:
            return {
                "status": "ERROR",
                "reason": "At least one seam_id is required.",
                "partial_plan_retained": False,
            }
        if len(set(seam_ids)) != len(seam_ids):
            return {
                "status": "ERROR",
                "reason": "Duplicate seam_ids detected. Each seam must be requested once.",
                "partial_plan_retained": False,
            }
        if not (0.0 < weld_speed_mm_s <= MAX_WELD_SPEED_MM_S):
            return {
                "status": "ERROR",
                "reason": f"weld_speed_mm_s must be in (0, {MAX_WELD_SPEED_MM_S}] mm/s.",
                "partial_plan_retained": False,
            }
        if not (0.0 <= approach_clearance_mm <= MAX_APPROACH_CLEARANCE_MM):
            return {
                "status": "ERROR",
                "reason": f"approach_clearance_mm must be in [0, {MAX_APPROACH_CLEARANCE_MM}] mm.",
                "partial_plan_retained": False,
            }

        try:
            arm = create_manipulator(robot_type)
        except ValueError as exc:
            return {
                "status": "ERROR",
                "reason": str(exc),
                "partial_plan_retained": False,
            }

        planned_waypoints: list[TrajectoryWaypoint] = []
        home_xyz = np.array([450.0, 0.0, 420.0], dtype=np.float64)
        home_tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        home_tool_orientation = arm.make_tool_orientation(home_tool_z)

        home_ik_success, q_home, home_err = arm.inverse_kinematics(
            target_pos=home_xyz,
            target_rot=home_tool_orientation,
        )
        if not home_ik_success:
            return {
                "status": "IK_UNREACHABLE",
                "robot_type": arm.robot_type,
                "waypoint": "HOME_SAFE",
                "target_xyz": home_xyz.tolist(),
                "target_tool_z": home_tool_z.tolist(),
                "position_error_mm": round(home_err, 3),
                "exceeded_coarse_reach": bool(np.linalg.norm(home_xyz) > arm.max_reach_mm),
                "partial_plan_retained": False,
            }

        planned_waypoints.append(
            TrajectoryWaypoint(
                label="HOME_SAFE",
                move_type="MOVJ",
                xyz_mm=home_xyz,
                joints_deg=np.rad2deg(q_home),
                speed_mm_s=150.0,
            )
        )
        q_current = q_home.copy()

        for seam_idx, seam_id in enumerate(seam_ids):
            if seam_id not in scene.seams:
                return {
                    "status": "ERROR",
                    "reason": (
                        f"Unknown seam_id '{seam_id}'. "
                        f"Available seam IDs: {list(scene.seams.keys())}."
                    ),
                    "partial_plan_retained": False,
                }

            # Safe overhead clearance transit between consecutive seams on different faces
            if seam_idx > 0:
                transit_coll = _check_joint_space_segment_collisions(
                    arm=arm,
                    q_start=q_current,
                    q_end=q_home,
                    segment_label=f"{seam_id}_SAFE_TRANSIT",
                    robot_type=arm.robot_type,
                )
                if transit_coll is not None:
                    return transit_coll

                planned_waypoints.append(
                    TrajectoryWaypoint(
                        label=f"{seam_id}_SAFE_TRANSIT",
                        move_type="MOVJ",
                        xyz_mm=home_xyz,
                        joints_deg=np.rad2deg(q_home),
                        speed_mm_s=150.0,
                    )
                )
                q_current = q_home.copy()
            
            seam = scene.seams[seam_id]
            seam_start = np.asarray(seam.start_xyz, dtype=np.float64)
            seam_end = np.asarray(seam.end_xyz, dtype=np.float64)
            seam_length = float(np.linalg.norm(seam_end - seam_start))
            if seam_length < 1e-6:
                return {
                    "status": "ERROR",
                    "reason": f"Seam '{seam_id}' has zero length (start_xyz == end_xyz).",
                    "partial_plan_retained": False,
                }

            raw_normal = np.asarray(seam.normal_xyz, dtype=np.float64)
            normal_length = float(np.linalg.norm(raw_normal))
            if normal_length < 1e-9:
                return {
                    "status": "ERROR",
                    "reason": f"Seam '{seam_id}' has an invalid zero-length normal.",
                    "partial_plan_retained": False,
                }
            seam_normal = raw_normal / normal_length

            approach_xyz = seam_start + seam_normal * approach_clearance_mm
            retract_xyz = seam_end + seam_normal * approach_clearance_mm
            target_tool_z = -seam_normal if tool_z_points_into_workpiece else seam_normal
            tool_orientation = arm.make_tool_orientation(target_tool_z)

            # 1. Solve and continuously check MOVJ approach segment (q_current -> q_approach)
            ok_app, q_approach, err_app = arm.inverse_kinematics(
                target_pos=approach_xyz,
                target_rot=tool_orientation,
                q_init=q_current,
            )
            if not ok_app:
                return {
                    "status": "IK_UNREACHABLE",
                    "robot_type": arm.robot_type,
                    "waypoint": f"{seam_id}_APPROACH",
                    "target_xyz": approach_xyz.tolist(),
                    "target_tool_z": target_tool_z.tolist(),
                    "position_error_mm": round(err_app, 3),
                    "exceeded_coarse_reach": bool(np.linalg.norm(approach_xyz) > arm.max_reach_mm),
                    "partial_plan_retained": False,
                }

            if arm.joint_jump(q_approach, q_current) > MAX_WAYPOINT_JOINT_JUMP_DEGREES:
                return {
                    "status": "JOINT_DISCONTINUITY",
                    "robot_type": arm.robot_type,
                    "waypoint": f"{seam_id}_APPROACH",
                    "details": f"Jump {arm.joint_jump(q_approach, q_current):.1f} deg exceeds limit.",
                    "partial_plan_retained": False,
                }

            movj_err = _check_joint_space_segment_collisions(
                arm=arm,
                q_start=q_current,
                q_end=q_approach,
                segment_label=f"{seam_id}_TO_APPROACH",
                robot_type=arm.robot_type,
            )
            if movj_err is not None:
                return movj_err

            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_APPROACH",
                    move_type="MOVJ",
                    xyz_mm=approach_xyz,
                    joints_deg=np.rad2deg(q_approach),
                                        speed_mm_s=120.0,
                )
            )
            q_current = q_approach.copy()

            # 2. Linear plunge from APPROACH to START (MOVL)
            ok_start, q_start, err_start = arm.inverse_kinematics(
                target_pos=seam_start,
                target_rot=tool_orientation,
                q_init=q_current,
            )
            if not ok_start:
                return {
                    "status": "IK_UNREACHABLE",
                    "robot_type": arm.robot_type,
                    "waypoint": f"{seam_id}_START",
                    "target_xyz": seam_start.tolist(),
                    "target_tool_z": target_tool_z.tolist(),
                    "position_error_mm": round(err_start, 3),
                    "exceeded_coarse_reach": bool(
                        np.linalg.norm(seam_start) > arm.max_reach_mm
                    ),
                    "partial_plan_retained": False,
                }

            start_coll, start_msg = scene.check_chain_collision(
                arm.forward_kinematics_chain(q_start)
            )
            if start_coll:
                return {
                    "status": "COLLISION_DETECTED",
                    "robot_type": arm.robot_type,
                    "waypoint": f"{seam_id}_START",
                    "details": start_msg,
                    "partial_plan_retained": False,
                }

            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_START",
                    move_type="MOVL",
                    xyz_mm=seam_start,
                    joints_deg=np.rad2deg(q_start),
                    arc_state="ARC_ON",
                    speed_mm_s=weld_speed_mm_s,
                )
            )
            q_current = q_start.copy()

            # 3. Uniform dense Cartesian sampling along the weld seam with sequential q_init tracking
            weld_samples = np.linspace(
                seam_start, seam_end, WELD_PATH_COLLISION_SAMPLES
            )
            q_prev_sample = q_start.copy()
            q_mid = q_start.copy()
            q_end = q_start.copy()
            mid_index = WELD_PATH_COLLISION_SAMPLES // 2

            for sample_idx, sample_xyz in enumerate(weld_samples):
                s_ok, s_q, s_err = arm.inverse_kinematics(
                    target_pos=sample_xyz,
                    target_rot=tool_orientation,
                    q_init=q_prev_sample,
                )
                if not s_ok:
                    return {
                        "status": "IK_UNREACHABLE",
                        "robot_type": arm.robot_type,
                        "waypoint": f"{seam_id}_WELD_SAMPLE_{sample_idx}",
                        "target_xyz": sample_xyz.tolist(),
                        "target_tool_z": target_tool_z.tolist(),
                        "position_error_mm": round(s_err, 3),
                        "exceeded_coarse_reach": bool(
                            np.linalg.norm(sample_xyz) > arm.max_reach_mm
                        ),
                        "partial_plan_retained": False,
                    }

                sample_jump = arm.joint_jump(s_q, q_prev_sample)
                if sample_jump > MAX_SAMPLE_JOINT_JUMP_DEGREES:
                    return {
                        "status": "JOINT_DISCONTINUITY",
                        "robot_type": arm.robot_type,
                        "waypoint": f"{seam_id}_WELD_SAMPLE_{sample_idx}",
                        "details": (
                            f"Branch flip / excessive joint jump ({sample_jump:.1f} deg) "
                            f"between adjacent weld samples."
                        ),
                        "partial_plan_retained": False,
                    }

                s_coll, s_msg = scene.check_chain_collision(
                    arm.forward_kinematics_chain(s_q)
                )
                if s_coll:
                    return {
                        "status": "COLLISION_DETECTED",
                        "robot_type": arm.robot_type,
                        "waypoint": f"{seam_id}_WELD_SAMPLE_{sample_idx}",
                        "details": s_msg,
                        "partial_plan_retained": False,
                    }

                if sample_idx == mid_index:
                    q_mid = s_q.copy()
                if sample_idx == WELD_PATH_COLLISION_SAMPLES - 1:
                    q_end = s_q.copy()

                q_prev_sample = s_q.copy()

            mid_xyz = 0.5 * (seam_start + seam_end)
            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_MID",
                    move_type="MOVL",
                    xyz_mm=mid_xyz,
                    joints_deg=np.rad2deg(q_mid),
                    speed_mm_s=weld_speed_mm_s,
                )
            )
            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_END",
                    move_type="MOVL",
                    xyz_mm=seam_end,
                    joints_deg=np.rad2deg(q_end),
                    arc_state="ARC_OFF",
                    speed_mm_s=weld_speed_mm_s,
                )
            )
            q_current = q_end.copy()

            # 4. Solve and continuously check retract segment (q_end -> q_retract)
            ok_ret, q_retract, err_ret = arm.inverse_kinematics(
                target_pos=retract_xyz,
                target_rot=tool_orientation,
                q_init=q_current,
            )
            if not ok_ret:
                return {
                    "status": "IK_UNREACHABLE",
                    "robot_type": arm.robot_type,
                    "waypoint": f"{seam_id}_RETRACT",
                    "target_xyz": retract_xyz.tolist(),
                    "target_tool_z": target_tool_z.tolist(),
                    "position_error_mm": round(err_ret, 3),
                    "exceeded_coarse_reach": bool(
                        np.linalg.norm(retract_xyz) > arm.max_reach_mm
                    ),
                    "partial_plan_retained": False,
                }

            ret_err = _check_joint_space_segment_collisions(
                arm=arm,
                q_start=q_current,
                q_end=q_retract,
                segment_label=f"{seam_id}_TO_RETRACT",
                robot_type=arm.robot_type,
            )
            if ret_err is not None:
                return ret_err

            planned_waypoints.append(
                TrajectoryWaypoint(
                    label=f"{seam_id}_RETRACT",
                    move_type="MOVJ",
                    xyz_mm=retract_xyz,
                    joints_deg=np.rad2deg(q_retract),
                    speed_mm_s=120.0,
                )
            )
            q_current = q_retract.copy()

        # 5. Return to HOME_SAFE with continuous joint-space collision check
        ok_ret_home, q_ret_home, err_ret_home = arm.inverse_kinematics(
            target_pos=home_xyz,
            target_rot=home_tool_orientation,
            q_init=q_current,
        )
        if not ok_ret_home:
            return {
                "status": "IK_UNREACHABLE",
                "robot_type": arm.robot_type,
                "waypoint": "RETURN_HOME",
                "target_xyz": home_xyz.tolist(),
                "target_tool_z": home_tool_z.tolist(),
                "position_error_mm": round(err_ret_home, 3),
                "exceeded_coarse_reach": False,
                "partial_plan_retained": False,
            }

        ret_home_coll = _check_joint_space_segment_collisions(
            arm=arm,
            q_start=q_current,
            q_end=q_ret_home,
            segment_label="RETURN_HOME",
            robot_type=arm.robot_type,
        )
        if ret_home_coll is not None:
            return ret_home_coll

        planned_waypoints.append(
            TrajectoryWaypoint(
                label="RETURN_HOME",
                move_type="MOVJ",
                xyz_mm=home_xyz,
                joints_deg=np.rad2deg(q_ret_home),
                speed_mm_s=150.0,
            )
        )

        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        record = ValidatedPlanRecord(
            plan_id=plan_id,
            robot_type=arm.robot_type,
            dof=arm.dof,
            seams_planned=list(seam_ids),
            waypoints=tuple(planned_waypoints),
        )
        validated_plans[plan_id] = record
        latest_plan_id = plan_id

        return {
            "status": "VALIDATED_OK",
            "validation_level": "kinematic_and_continuous_static_collision",
            "plan_id": plan_id,
            "robot_type": arm.robot_type,
            "dof": arm.dof,
            "seams_planned": list(seam_ids),
            "total_waypoints": len(planned_waypoints),
            "collision_free": True,
            "weld_path_samples_per_seam": WELD_PATH_COLLISION_SAMPLES,
            "movj_samples_per_segment": MOVJ_COLLISION_SAMPLES,
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

    except np.linalg.LinAlgError as exc:
        return {
            "status": "IK_SOLVER_ERROR",
            "reason": f"Linear algebra failure during IK: {exc}",
            "partial_plan_retained": False,
        }
    except Exception as exc:
        logger.exception("Weld planning failed unexpectedly")
        return {
            "status": "INTERNAL_ERROR",
            "reason": str(exc),
            "partial_plan_retained": False,
        }


@mcp.tool()
def export_robot_program(
    dialect: str = "melfa",
    program_name: str = "WELD_JOB_01",
    plan_id: str | None = None,
) -> dict:
    """
    Compiles a validated trajectory (by plan_id or latest validated plan)
    into vendor-specific robot controller code.
    """
    target_id = plan_id if plan_id is not None else latest_plan_id
    if target_id is None or target_id not in validated_plans:
        return {
            "status": "ERROR",
            "reason": (
                "No matching validated trajectory found. "
                "Call plan_and_simulate_weld first or pass a valid plan_id."
            ),
        }

    sanitized_name = program_name.strip()
    if not PROGRAM_NAME_PATTERN.match(sanitized_name):
        return {
            "status": "ERROR",
            "reason": (
                "Invalid program_name. Use 1-64 alphanumeric characters, "
                "underscores, or hyphens."
            ),
        }

    normalized_dialect = dialect.strip().lower()
    if normalized_dialect not in SUPPORTED_PROGRAM_DIALECTS:
        return {
            "status": "ERROR",
            "reason": (
                f"Unsupported dialect '{dialect}'. "
                f"Supported dialects: {sorted(SUPPORTED_PROGRAM_DIALECTS)}."
            ),
        }

    record = validated_plans[target_id]
    program_code = compile_program(
        list(record.waypoints),
        dialect=normalized_dialect,
        program_name=sanitized_name,
        robot_type=record.robot_type,
    )
    return {
        "status": "COMPILED",
        "plan_id": record.plan_id,
        "robot_type": record.robot_type,
        "dialect": normalized_dialect,
        "line_count": len(program_code.splitlines()),
        "program_code": program_code,
    }


if __name__ == "__main__":
    mcp.run()
