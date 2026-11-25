from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

Vector3 = list[float]


def _as_vector3(value: object, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{field_name} must contain exactly three coordinates.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain only finite values.")

    return vector


def _frozen_vector3(value: np.ndarray) -> np.ndarray:
    value = np.array(value, dtype=np.float64, copy=True)
    value.flags.writeable = False
    return value


class WeldSeam(BaseModel):
    """A straight weld seam defined in the robot/world base frame."""

    seam_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Unique identifier of the weld seam.",
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable seam description.",
    )

    start_xyz: Vector3 = Field(
        ...,
        description="Seam start point [x, y, z] in mm.",
    )

    end_xyz: Vector3 = Field(
        ...,
        description="Seam end point [x, y, z] in mm.",
    )

    normal_xyz: Vector3 = Field(
        default=[0.0, 0.0, 1.0],
        description=(
            "Outward workpiece-surface normal at the seam. "
            "It is normalized automatically."
        ),
    )

    weld_type: Literal["fillet", "butt", "lap", "spot"] = Field(
        default="fillet",
        description="Weld joint type.",
    )

    @field_validator("start_xyz", "end_xyz", "normal_xyz")
    @classmethod
    def validate_vector(cls, value: Vector3, info) -> Vector3:
        _as_vector3(value, info.field_name)
        return value

    @model_validator(mode="after")
    def validate_and_normalize(self) -> WeldSeam:
        start = _as_vector3(self.start_xyz, "start_xyz")
        end = _as_vector3(self.end_xyz, "end_xyz")
        normal = _as_vector3(self.normal_xyz, "normal_xyz")

        seam_length_mm = float(np.linalg.norm(end - start))
        if seam_length_mm < 1.0:
            raise ValueError(
                f"Seam '{self.seam_id}' must be at least 1.0 mm long; "
                f"got {seam_length_mm:.3f} mm."
            )

        normal_length = float(np.linalg.norm(normal))
        if normal_length < 1e-9:
            raise ValueError(
                f"Seam '{self.seam_id}' has an invalid zero-length normal."
            )

        object.__setattr__(self, "normal_xyz", (normal / normal_length).tolist())
        return self

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_xyz, dtype=np.float64)

    @property
    def end(self) -> np.ndarray:
        return np.asarray(self.end_xyz, dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self.normal_xyz, dtype=np.float64)

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def path_points(self, samples: int) -> np.ndarray:
        """Returns uniformly spaced points along the straight seam."""
        if samples < 2:
            raise ValueError("samples must be at least 2.")

        return np.linspace(self.start, self.end, samples)


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box with an optional collision margin."""

    name: str
    min_pt: np.ndarray
    max_pt: np.ndarray
    collision_margin_mm: float = 5.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AABB name must not be empty.")

        min_pt = _as_vector3(self.min_pt, "min_pt")
        max_pt = _as_vector3(self.max_pt, "max_pt")

        if np.any(min_pt > max_pt):
            raise ValueError(f"AABB '{self.name}': min_pt must not exceed max_pt.")

        if self.collision_margin_mm < 0.0:
            raise ValueError(
                f"AABB '{self.name}': collision_margin_mm must be non-negative."
            )

        object.__setattr__(self, "min_pt", _frozen_vector3(min_pt))
        object.__setattr__(self, "max_pt", _frozen_vector3(max_pt))

    def contains_point(
        self,
        point: np.ndarray,
        margin: float = 0.0,
    ) -> bool:
        point = _as_vector3(point, "point")

        # if margin < 0.0:
        #    raise ValueError("margin must be non-negative.")

        lower = self.min_pt - margin
        upper = self.max_pt + margin

        return bool(np.all(point >= lower) and np.all(point <= upper))

    def intersects_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        margin: float = 0.0,
    ) -> bool:
        """
        Exact segment-versus-expanded-AABB intersection using the slab method.

        A positive margin expands the box and is useful for capsule-like
        clearance around a robot link or tool.
        """
        start = _as_vector3(start, "start")
        end = _as_vector3(end, "end")

        if margin < 0.0:
            raise ValueError("margin must be non-negative.")

        lower = self.min_pt - margin
        upper = self.max_pt + margin
        delta = end - start

        t_min = 0.0
        t_max = 1.0

        for axis in range(3):
            if abs(float(delta[axis])) < 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue

            inverse_delta = 1.0 / float(delta[axis])
            t_enter = (lower[axis] - start[axis]) * inverse_delta
            t_exit = (upper[axis] - start[axis]) * inverse_delta

            if t_enter > t_exit:
                t_enter, t_exit = t_exit, t_enter

            t_min = max(t_min, t_enter)
            t_max = min(t_max, t_exit)

            if t_min > t_max:
                return False

        return True


@dataclass(frozen=True)
class LinkGeometry:
    """
    Conservative collision geometry for one robot link.

    The link is approximated by a capsule between two kinematic-frame origins.
    `radius_mm` must include the physical link envelope; it is not a safety
    margin.
    """

    name: str
    start_frame_index: int
    end_frame_index: int
    radius_mm: float
    collision_margin_mm: float = 5.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LinkGeometry name must not be empty.")

        if self.start_frame_index < 0 or self.end_frame_index <= self.start_frame_index:
            raise ValueError(
                f"LinkGeometry '{self.name}': end_frame_index must be greater "
                "than start_frame_index."
            )

        if self.radius_mm <= 0.0:
            raise ValueError(f"LinkGeometry '{self.name}': radius_mm must be positive.")

        if self.collision_margin_mm < 0.0:
            raise ValueError(
                f"LinkGeometry '{self.name}': collision_margin_mm must be "
                "non-negative."
            )


@dataclass(frozen=True)
class CollisionResult:
    """Structured result of a static robot-configuration collision check."""

    collided: bool
    message: str
    obstacle_name: str | None = None
    link_name: str | None = None
    sample_point_mm: np.ndarray | None = None
    clearance_mm: float | None = None


class WorkpieceScene:
    """
    Represents the robot mounting pose, collision environment, workpiece,
    and weld seams.

    All geometry is expressed in the robot/world base frame in millimetres.
    The default robot base is the identity transform; supply a different
    `robot_base_transform` when the physical robot is not mounted at the
    world origin.
    """

    def __init__(
        self,
        robot_base_transform: np.ndarray | None = None,
    ) -> None:
        if robot_base_transform is None:
            robot_base_transform = np.eye(4, dtype=np.float64)

        base = np.asarray(robot_base_transform, dtype=np.float64)
        if base.shape != (4, 4) or not np.all(np.isfinite(base)):
            raise ValueError(
                "robot_base_transform must be a finite 4x4 homogeneous matrix."
            )

        if not np.allclose(base[3, :], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("robot_base_transform must have bottom row [0, 0, 0, 1].")

        self.robot_base_transform = base.copy()
        self.robot_base_transform.flags.writeable = False

        self.part_name = "Bracket_Assembly_REV_B"

        self.obstacles: list[AABB] = [
            AABB(
                name="Welding_Table",
                min_pt=np.array([-1000.0, -1000.0, -200.0]),
                max_pt=np.array([1500.0, 1000.0, -1.0]),
                collision_margin_mm=0.0,
            ),
            AABB(
                name="Fixture_Clamp_Left",
                min_pt=np.array([430.0, -180.0, 0.0]),
                max_pt=np.array([470.0, -120.0, 120.0]),
                collision_margin_mm=10.0,
            ),
            AABB(
                name="Fixture_Clamp_Right",
                min_pt=np.array([430.0, 120.0, 0.0]),
                max_pt=np.array([470.0, 180.0, 120.0]),
                collision_margin_mm=10.0,
            ),
        ]

        self.workpiece_obstacles: list[AABB] = [
            AABB(
                name="Workpiece_Core_Body",
                min_pt=np.array([460.0, -140.0, 0.0]),
                max_pt=np.array([740.0, 140.0, 175.0]),
                collision_margin_mm=5.0,
            ),
        ]

        self.seams: dict[str, WeldSeam] = {
            seam.seam_id: seam
            for seam in (
                WeldSeam(
                    seam_id="seam_top_rib",
                    description="Top longitudinal rib weld along the upper plate.",
                    start_xyz=[480.0, 0.0, 180.0],
                    end_xyz=[720.0, 0.0, 180.0],
                    normal_xyz=[0.0, 0.0, 1.0],
                    weld_type="fillet",
                ),
                WeldSeam(
                    seam_id="seam_left_flange",
                    description="Left side flange fillet weld.",
                    start_xyz=[500.0, -145.0, 90.0],
                    end_xyz=[700.0, -145.0, 90.0],
                    normal_xyz=[0.0, -0.7071067811865476, 0.7071067811865476],
                    weld_type="fillet",
                ),
                WeldSeam(
                    seam_id="seam_right_flange",
                    description="Right side flange fillet weld.",
                    start_xyz=[500.0, 145.0, 90.0],
                    end_xyz=[700.0, 145.0, 90.0],
                    normal_xyz=[0.0, 0.7071067811865476, 0.7071067811865476],
                    weld_type="fillet",
                ),
            )
        }

        self.link_geometries: list[LinkGeometry] = [
            LinkGeometry(
                name="upper_arm",
                start_frame_index=1,
                end_frame_index=2,
                radius_mm=45.0,
                collision_margin_mm=10.0,
            ),
            LinkGeometry(
                name="forearm",
                start_frame_index=2,
                end_frame_index=3,
                radius_mm=35.0,
                collision_margin_mm=10.0,
            ),
            LinkGeometry(
                name="wrist",
                start_frame_index=3,
                end_frame_index=4,
                radius_mm=30.0,
                collision_margin_mm=10.0,
            ),
            LinkGeometry(
                name="tool_body",
                start_frame_index=4,
                end_frame_index=5,
                radius_mm=18.0,
                collision_margin_mm=8.0,
            ),
        ]

    def _to_world_frames(
        self,
        joint_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        world_frames: list[np.ndarray] = []

        for frame in joint_frames:
            frame_array = np.asarray(frame, dtype=np.float64)

            if frame_array.shape != (4, 4):
                raise ValueError("Each joint frame must be a 4x4 homogeneous matrix.")

            if not np.all(np.isfinite(frame_array)):
                raise ValueError("Each joint frame must contain only finite values.")

            world_frames.append(self.robot_base_transform @ frame_array)

        return world_frames

    def get_safe_transit_position(
        self,
        robot_type: str,
    ) -> np.ndarray:
        if robot_type == "scara_4dof":
            return np.array(
                [450.0, 0.0, 430.0],
                dtype=np.float64,
            )

        return np.array(
            [450.0, 0.0, 650.0],
            dtype=np.float64,
        )

    def check_point_collision(
        self,
        point: np.ndarray,
        *,
        allow_workpiece_contact: bool = False,
        tool_contact_clearance_mm: float = 0.0,
    ) -> CollisionResult:
        """
        Checks one world-frame point against environment and workpiece volumes.

        `allow_workpiece_contact=True` permits the tool contact point to lie on
        the workpiece surface, but it does not permit the point to penetrate
        the workpiece volume.
        """
        point = _as_vector3(point, "point")

        for obstacle in self.obstacles:
            if obstacle.contains_point(point, obstacle.collision_margin_mm):
                return CollisionResult(
                    collided=True,
                    message=(
                        f"Point collides with environment obstacle "
                        f"'{obstacle.name}'."
                    ),
                    obstacle_name=obstacle.name,
                    sample_point_mm=point,
                )

        for obstacle in self.workpiece_obstacles:
            margin = obstacle.collision_margin_mm

            if allow_workpiece_contact:
                # Permit contact only at the boundary, not inside the solid.
                expanded_contains = obstacle.contains_point(
                    point,
                    margin=margin,
                )
                solid_contains = obstacle.contains_point(
                    point,
                    margin=-min(
                        tool_contact_clearance_mm,
                        margin,
                    ),
                )

                if expanded_contains and solid_contains:
                    return CollisionResult(
                        collided=True,
                        message=(
                            f"Tool contact point penetrates workpiece "
                            f"'{obstacle.name}'."
                        ),
                        obstacle_name=obstacle.name,
                        sample_point_mm=point,
                    )
            elif obstacle.contains_point(point, margin=margin):
                return CollisionResult(
                    collided=True,
                    message=(f"Point collides with workpiece " f"'{obstacle.name}'."),
                    obstacle_name=obstacle.name,
                    sample_point_mm=point,
                )

        return CollisionResult(collided=False, message="OK")

    def check_chain_collision(
        self,
        joint_frames: list[np.ndarray],
        *,
        allow_tool_contact: bool = False,
        tool_contact_clearance_mm: float = 0.0,
    ) -> CollisionResult:
        """
        Performs a conservative static collision check for a robot configuration.

        The robot base-to-first-joint segment is excluded because the base is
        assumed to be mounted outside the checked environment.

        Important limitation: this checks capsule-approximated links against
        AABB obstacles. It does not perform mesh-accurate collision detection,
        self-collision checking, or swept-motion collision checking.
        """
        if not joint_frames:
            return CollisionResult(
                collided=False,
                message="No joint frames provided.",
            )

        if len(joint_frames) < 2:
            return CollisionResult(
                collided=False,
                message="Not enough joint frames for collision checking.",
            )

        world_frames = self._to_world_frames(joint_frames)
        last_frame_index = len(world_frames) - 1

        for geometry in self.link_geometries:
            start_index = geometry.start_frame_index
            end_index = geometry.end_frame_index

            if end_index >= len(world_frames):
                return CollisionResult(
                    collided=False,
                    message=(
                        f"Link geometry '{geometry.name}' refers to frame "
                        f"{end_index}, but only {len(world_frames)} frames "
                        "were provided."
                    ),
                )

            start_point = world_frames[start_index][:3, 3]
            end_point = world_frames[end_index][:3, 3]

            clearance = geometry.radius_mm + geometry.collision_margin_mm

            for obstacle in self.obstacles:
                if obstacle.intersects_segment(
                    start_point,
                    end_point,
                    margin=clearance,
                ):
                    return CollisionResult(
                        collided=True,
                        message=(
                            f"Robot link '{geometry.name}' collides with "
                            f"'{obstacle.name}'."
                        ),
                        obstacle_name=obstacle.name,
                        link_name=geometry.name,
                        clearance_mm=clearance,
                    )

            workpiece_margin = geometry.radius_mm + geometry.collision_margin_mm

            is_tool_segment = (
                geometry.name == "tool_body"
                or geometry.end_frame_index == last_frame_index
            )

            if is_tool_segment and allow_tool_contact:
                # The tool contact point may approach the workpiece, but the
                # capsule must not penetrate the workpiece solid.
                workpiece_margin = 0.0

            for obstacle in self.workpiece_obstacles:
                if obstacle.intersects_segment(
                    start_point,
                    end_point,
                    margin=workpiece_margin,
                ):
                    return CollisionResult(
                        collided=True,
                        message=(
                            f"Robot link '{geometry.name}' collides with "
                            f"workpiece '{obstacle.name}'."
                        ),
                        obstacle_name=obstacle.name,
                        link_name=geometry.name,
                        clearance_mm=workpiece_margin,
                    )

        tcp_point = world_frames[last_frame_index][:3, 3]
        tcp_result = self.check_point_collision(
            tcp_point,
            allow_workpiece_contact=allow_tool_contact,
            tool_contact_clearance_mm=tool_contact_clearance_mm,
        )

        if tcp_result.collided:
            return CollisionResult(
                collided=True,
                message=f"TCP collision: {tcp_result.message}",
                obstacle_name=tcp_result.obstacle_name,
                link_name="tcp",
                sample_point_mm=tcp_point,
            )

        return CollisionResult(
            collided=False,
            message="OK",
        )

    def check_trajectory_collision(
        self,
        joint_frames_start: list[np.ndarray],
        joint_frames_end: list[np.ndarray],
        *,
        samples: int = 25,
        allow_tool_contact: bool = False,
        tool_contact_clearance_mm: float = 0.0,
    ) -> CollisionResult:
        """
        Checks a conservative joint-space interpolation between two configurations.

        This is appropriate for `MOVJ` motions. For Cartesian `MOVL` motions,
        interpolate TCP positions and solve IK at each sample instead.
        """
        if samples < 2:
            raise ValueError("samples must be at least 2.")

        if len(joint_frames_start) != len(joint_frames_end):
            raise ValueError("Start and end joint-frame lists must have equal length.")

        for alpha in np.linspace(0.0, 1.0, samples):
            interpolated_frames: list[np.ndarray] = []

            for start_frame, end_frame in zip(
                joint_frames_start,
                joint_frames_end,
            ):
                start_frame = np.asarray(start_frame, dtype=np.float64)
                end_frame = np.asarray(end_frame, dtype=np.float64)

                interpolated_frames.append(
                    (1.0 - alpha) * start_frame + alpha * end_frame
                )

            result = self.check_chain_collision(
                interpolated_frames,
                allow_tool_contact=allow_tool_contact,
                tool_contact_clearance_mm=tool_contact_clearance_mm,
            )

            if result.collided:
                return CollisionResult(
                    collided=True,
                    message=(
                        f"{result.message} "
                        f"(joint-space interpolation alpha={alpha:.3f})"
                    ),
                    obstacle_name=result.obstacle_name,
                    link_name=result.link_name,
                    sample_point_mm=result.sample_point_mm,
                )

        return CollisionResult(collided=False, message="OK")

    def check_cartesian_trajectory_collision(
        self,
        arm: Any,
        q_start: np.ndarray,
        target_positions: np.ndarray,
        target_orientation: np.ndarray,
        *,
        samples: int = 25,
        allow_tool_contact: bool = False,
        tool_contact_clearance_mm: float = 0.0,
        max_joint_jump_deg: float = 15.0,
        start_tolerance_mm: float = 0.5,
    ) -> tuple[CollisionResult, np.ndarray | None]:
        """
        Checks a Cartesian MOVL trajectory for reachability, continuity,
        and static collision.

        `target_positions` must be an (N, 3) array of TCP positions in mm.
        The first position should normally equal the TCP position produced by
        `q_start`.
        """
        if samples < 2:
            raise ValueError("samples must be at least 2.")

        positions = np.asarray(
            target_positions,
            dtype=np.float64,
        )

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("target_positions must have shape (N, 3).")

        if positions.shape[0] < 2:
            raise ValueError("target_positions must contain at least two positions.")

        if not np.all(np.isfinite(positions)):
            raise ValueError("target_positions must contain only finite values.")

        orientation = np.asarray(
            target_orientation,
            dtype=np.float64,
        )

        if orientation.shape != (3, 3) or not np.all(np.isfinite(orientation)):
            raise ValueError("target_orientation must be a finite 3x3 rotation matrix.")

        q_previous = np.asarray(
            q_start,
            dtype=np.float64,
        ).copy()

        if q_previous.shape != (arm.dof,):
            raise ValueError(f"q_start must have shape ({arm.dof},).")

        previous_frames = arm.forward_kinematics_chain(q_previous)
        previous_tcp = previous_frames[-1][:3, 3].copy()

        start_mismatch_mm = float(np.linalg.norm(previous_tcp - positions[0]))

        if start_mismatch_mm > start_tolerance_mm:
            return (
                CollisionResult(
                    collided=False,
                    message=(
                        "The first target position does not match the TCP "
                        f"position of q_start by "
                        f"{start_mismatch_mm:.3f} mm."
                    ),
                ),
                None,
            )

        positions = positions.copy()
        positions[0] = previous_tcp

        for sample_index, target_position in enumerate(positions):
            ik_success, q_sample, position_error = arm.inverse_kinematics(
                target_pos=target_position,
                target_rot=orientation,
                q_init=q_previous,
                max_iters=800,
                pos_tol=0.2,
                max_joint_jump_deg=max_joint_jump_deg,
            )

            if not ik_success:
                return (
                    CollisionResult(
                        collided=True,
                        message=(
                            f"Cartesian trajectory sample {sample_index} is "
                            f"unreachable; position error "
                            f"{position_error:.3f} mm."
                        ),
                        sample_point_mm=target_position.copy(),
                    ),
                    None,
                )

            joint_jump = arm.joint_jump(
                q_sample,
                q_previous,
            )

            if joint_jump > max_joint_jump_deg:
                return (
                    CollisionResult(
                        collided=True,
                        message=(
                            f"Cartesian trajectory sample {sample_index} creates "
                            f"a joint-space discontinuity of "
                            f"{joint_jump:.3f} degrees."
                        ),
                        sample_point_mm=target_position.copy(),
                    ),
                    None,
                )

            sample_frames = arm.forward_kinematics_chain(q_sample)

            collision_result = self.check_chain_collision(
                sample_frames,
                allow_tool_contact=allow_tool_contact,
                tool_contact_clearance_mm=tool_contact_clearance_mm,
            )

            if collision_result.collided:
                return (
                    CollisionResult(
                        collided=True,
                        message=(
                            f"{collision_result.message} "
                            f"(Cartesian trajectory sample {sample_index})"
                        ),
                        obstacle_name=collision_result.obstacle_name,
                        link_name=collision_result.link_name,
                        sample_point_mm=target_position.copy(),
                    ),
                    None,
                )

            q_previous = q_sample.copy()

        return CollisionResult(collided=False, message="OK"), q_previous.copy()

    def get_workpiece_metadata(self) -> dict:
        """Returns scene metadata suitable for an MCP resource."""
        return {
            "part_name": self.part_name,
            "robot_base_transform": self.robot_base_transform.tolist(),
            "obstacles": [
                {
                    "name": obstacle.name,
                    "min_pt_mm": obstacle.min_pt.tolist(),
                    "max_pt_mm": obstacle.max_pt.tolist(),
                    "collision_margin_mm": obstacle.collision_margin_mm,
                }
                for obstacle in self.obstacles
            ],
            "workpiece_obstacles": [
                {
                    "name": obstacle.name,
                    "min_pt_mm": obstacle.min_pt.tolist(),
                    "max_pt_mm": obstacle.max_pt.tolist(),
                    "collision_margin_mm": obstacle.collision_margin_mm,
                }
                for obstacle in self.workpiece_obstacles
            ],
            "seams": [seam.model_dump() for seam in self.seams.values()],
        }
