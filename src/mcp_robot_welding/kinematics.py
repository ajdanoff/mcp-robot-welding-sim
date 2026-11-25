from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .load_robots import (
    DEFAULT_ROBOT_CONFIG_DIR,
    build_supported_robots_map,
    load_robot_config_file,
    load_robot_configs,
    parse_angle_rad,
    parse_joint_limit,
)

logger = logging.getLogger("mcp_robot_welding.kinematics")


JointType = Literal["revolute", "prismatic"]

_ROBOT_CONFIG_REGISTRY: dict[str, dict[str, Any]] | None = None
SUPPORTED_ROBOTS: dict[str, str] = {}


def _get_robot_registry(
    config_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    global _ROBOT_CONFIG_REGISTRY

    if config_dir is not None:
        return load_robot_configs(config_dir)

    if _ROBOT_CONFIG_REGISTRY is None:
        _ROBOT_CONFIG_REGISTRY = load_robot_configs(DEFAULT_ROBOT_CONFIG_DIR)

    return _ROBOT_CONFIG_REGISTRY


def _refresh_supported_robots() -> dict[str, str]:
    SUPPORTED_ROBOTS.clear()
    SUPPORTED_ROBOTS.update(build_supported_robots_map(_get_robot_registry()))
    return SUPPORTED_ROBOTS


def reload_robot_configs(
    config_dir: Path | str = DEFAULT_ROBOT_CONFIG_DIR,
) -> dict[str, str]:
    """Reloads robot configs from disk and updates `SUPPORTED_ROBOTS`."""
    global _ROBOT_CONFIG_REGISTRY

    _ROBOT_CONFIG_REGISTRY = load_robot_configs(config_dir)

    SUPPORTED_ROBOTS.clear()
    SUPPORTED_ROBOTS.update(build_supported_robots_map(_ROBOT_CONFIG_REGISTRY))

    return SUPPORTED_ROBOTS


def _build_manipulator_from_config(
    config: dict[str, Any],
) -> RobotManipulator:
    """Instantiates a `RobotManipulator` from a validated config dictionary."""
    robot_type = str(config["robot_type"])
    dh_raw = config["dh_table"]
    limits_raw = config["joint_limits"]
    dof = len(dh_raw)

    dh_table: list[DHParameter] = [
        DHParameter(
            a=float(row["a"]),
            alpha=parse_angle_rad(row, "alpha", "alpha_deg", 0.0),
            d=float(row["d"]),
            theta_offset=parse_angle_rad(row, "theta_offset", "theta_offset_deg", 0.0),
            joint_type=str(row.get("joint_type", "revolute")),
        )
        for row in dh_raw
    ]

    joint_limits = [parse_joint_limit(limit_entry) for limit_entry in limits_raw]

    seeds_raw = config.get("seeds") or [[0.0] * dof]
    seeds: list[np.ndarray] = []

    for seed_index, seed in enumerate(seeds_raw):
        seed_array = np.asarray(seed, dtype=np.float64)

        if seed_array.shape != (dof,):
            raise ValueError(
                f"Config for '{robot_type}': seed {seed_index} shape "
                f"{seed_array.shape} does not match DOF ({dof})."
            )

        if not np.all(np.isfinite(seed_array)):
            raise ValueError(
                f"Config for '{robot_type}': seed {seed_index} "
                "contains non-finite values."
            )

        seeds.append(seed_array.copy())

    return RobotManipulator(
        robot_type=robot_type,
        description=str(config.get("description", robot_type)),
        dh_table=dh_table,
        joint_limits=joint_limits,
        seeds=seeds,
        max_reach_mm=float(config["max_reach_mm"]),
        position_only_orientation=bool(config.get("position_only_orientation", False)),
        tool_z_points_into_workpiece=bool(
            config.get("tool_z_points_into_workpiece", True)
        ),
    )


@dataclass(frozen=True)
class DHParameter:
    """One standard-DH transform.

    Lengths are in millimetres and angles are in radians.
    """

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0
    joint_type: JointType = "revolute"


class RobotManipulator:
    """Generic N-DoF industrial manipulator with revolute and prismatic joints."""

    def __init__(
        self,
        robot_type: str,
        description: str,
        dh_table: list[DHParameter],
        joint_limits: list[tuple[float, float]],
        seeds: list[np.ndarray],
        max_reach_mm: float,
        position_only_orientation: bool = False,
        tool_z_points_into_workpiece: bool = True,
    ) -> None:
        if not robot_type:
            raise ValueError("robot_type must not be empty.")
        if not description:
            raise ValueError("description must not be empty.")

        if len(dh_table) != len(joint_limits):
            raise ValueError("dh_table and joint_limits must have equal length.")

        if not dh_table:
            raise ValueError("dh_table must contain at least one joint.")

        if max_reach_mm <= 0.0:
            raise ValueError("max_reach_mm must be positive.")

        if not seeds:
            raise ValueError("At least one IK seed is required.")

        self.robot_type = robot_type
        self.description = description
        self.dh_table = list(dh_table)
        self.joint_limits = list(joint_limits)
        self.seeds = [np.asarray(seed, dtype=np.float64).copy() for seed in seeds]
        self.max_reach_mm = float(max_reach_mm)
        self.position_only_orientation = bool(position_only_orientation)
        self.tool_z_points_into_workpiece = bool(tool_z_points_into_workpiece)
        self.dof = len(self.dh_table)

        for index, seed in enumerate(self.seeds):
            if seed.shape != (self.dof,):
                raise ValueError(
                    f"Seed {index} must have shape ({self.dof},); " f"got {seed.shape}."
                )

            if not np.all(np.isfinite(seed)):
                raise ValueError(f"Seed {index} contains non-finite values.")

        for index, (lower, upper) in enumerate(self.joint_limits):
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"Joint limit {index} contains non-finite values.")

            if lower > upper:
                raise ValueError(
                    f"Joint limit {index}: lower bound exceeds upper bound."
                )

        for index, parameter in enumerate(self.dh_table):
            values = (parameter.a, parameter.alpha, parameter.d, parameter.theta_offset)

            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"DH parameter {index} contains non-finite values.")

    @staticmethod
    def _dh_transform(
        a: float,
        alpha: float,
        d: float,
        theta: float,
    ) -> np.ndarray:
        """Standard DH transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _validate_joint_vector(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)

        if joints.shape != (self.dof,):
            raise ValueError(
                f"joints must have shape ({self.dof},); got {joints.shape}."
            )

        if not np.all(np.isfinite(joints)):
            raise ValueError("joints must contain only finite values.")

        return joints

    def forward_kinematics_chain(
        self,
        joints: np.ndarray,
    ) -> list[np.ndarray]:
        """Returns homogeneous transforms for the base and every joint frame."""
        joints = self._validate_joint_vector(joints)

        frames: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        current_transform = np.eye(4, dtype=np.float64)

        for index, parameter in enumerate(self.dh_table):
            joint_value = float(joints[index])

            if parameter.joint_type == "prismatic":
                theta = parameter.theta_offset
                d = parameter.d + joint_value
            else:
                theta = joint_value + parameter.theta_offset
                d = parameter.d

            link_transform = self._dh_transform(
                parameter.a,
                parameter.alpha,
                d,
                theta,
            )

            current_transform = current_transform @ link_transform
            frames.append(current_transform.copy())

        return frames

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Returns the 4x4 SE(3) pose of the welding-tool TCP."""
        return self.forward_kinematics_chain(joints)[-1]

    def compute_jacobian_from_frames(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray:
        """Computes the 6xN geometric Jacobian from precomputed FK frames."""
        if len(frames) != self.dof + 1:
            raise ValueError(f"Expected {self.dof + 1} FK frames; got {len(frames)}.")

        tcp_position = frames[-1][:3, 3]
        jacobian = np.zeros((6, self.dof), dtype=np.float64)

        for index, parameter in enumerate(self.dh_table):
            joint_frame = frames[index]
            joint_axis = joint_frame[:3, 2]
            joint_origin = joint_frame[:3, 3]

            if parameter.joint_type == "prismatic":
                jacobian[:3, index] = joint_axis
                jacobian[3:, index] = 0.0
            else:
                jacobian[:3, index] = np.cross(
                    joint_axis,
                    tcp_position - joint_origin,
                )
                jacobian[3:, index] = joint_axis

        return jacobian

    def compute_jacobian(self, joints: np.ndarray) -> np.ndarray:
        """Computes the 6xN geometric Jacobian in the robot base frame."""
        frames = self.forward_kinematics_chain(joints)
        return self.compute_jacobian_from_frames(frames)

    @staticmethod
    def _rotation_error(
        current_rotation: np.ndarray,
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the rotation-vector error from current to target orientation.

        The result is expressed in the base frame, matching the angular rows
        of the geometric Jacobian.
        """
        current_rotation = np.asarray(current_rotation, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)

        if current_rotation.shape != (3, 3) or target_rotation.shape != (3, 3):
            raise ValueError("Rotation matrices must have shape (3, 3).")

        error_rotation = target_rotation @ current_rotation.T
        trace = float(np.trace(error_rotation))
        cosine_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cosine_angle))

        if angle < 1e-8:
            return np.zeros(3, dtype=np.float64)

        skew = np.array(
            [
                error_rotation[2, 1] - error_rotation[1, 2],
                error_rotation[0, 2] - error_rotation[2, 0],
                error_rotation[1, 0] - error_rotation[0, 1],
            ],
            dtype=np.float64,
        )

        sine_angle = float(np.sin(angle))

        # Near 180 degrees, use the symmetric part of R to recover the axis.
        if sine_angle < 1e-6:
            symmetric = (error_rotation + error_rotation.T) / 2.0
            axis = np.real(np.linalg.eigvals(symmetric))

            # Eigenvector associated with eigenvalue 1 is the rotation axis.
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            axis = eigenvectors[:, int(np.argmax(eigenvalues))]

            if np.dot(axis, skew) < 0.0:
                axis = -axis

            axis = axis / np.linalg.norm(axis)
            return axis * angle

        return (skew / (2.0 * sine_angle)) * angle

    def _wrap_revolute_joint(
        self,
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        # Only wrap joints whose physical range is strictly [-pi, pi]
        # if np.isclose(lower, -np.pi) and np.isclose(upper, np.pi):
        #     return (value + np.pi) % (2.0 * np.pi) - np.pi

        return value

    def _clip_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        joints = self._validate_joint_vector(joints)
        clipped = joints.copy()

        for index, (lower, upper) in enumerate(self.joint_limits):
            value = float(clipped[index])

            if self.dh_table[index].joint_type == "revolute":
                value = self._wrap_revolute_joint(value, lower, upper)

            clipped[index] = np.clip(value, lower, upper)

        return clipped

    def make_tool_orientation(
        self,
        tool_z_axis: np.ndarray,
        reference_x_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Builds a right-handed SO(3) matrix whose tool Z axis is requested."""
        tool_z_axis = _as_vector3(tool_z_axis, "tool_z_axis")

        norm = float(np.linalg.norm(tool_z_axis))
        if norm < 1e-9:
            raise ValueError("tool_z_axis must be non-zero.")

        tool_z_axis = tool_z_axis / norm

        if reference_x_axis is not None:
            proj = (
                reference_x_axis - np.dot(reference_x_axis, tool_z_axis) * tool_z_axis
            )
            if float(np.linalg.norm(proj)) > 1e-6:
                tool_x_axis = proj / np.linalg.norm(proj)
                tool_y_axis = np.cross(tool_z_axis, tool_x_axis)
                return np.column_stack([tool_x_axis, tool_y_axis, tool_z_axis])

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, tool_z_axis))) > 0.99:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        tool_x_axis = np.cross(helper, tool_z_axis)
        tool_x_axis = tool_x_axis / np.linalg.norm(tool_x_axis)
        tool_y_axis = np.cross(tool_z_axis, tool_x_axis)

        return np.column_stack([tool_x_axis, tool_y_axis, tool_z_axis])

    def default_tool_orientation(self) -> np.ndarray:
        """
        Returns the default torch orientation.

        For `tool_z_points_into_workpiece=True`, the tool Z axis points
        downward, along base -Z. This is the usual convention when the TCP
        Z axis points from the torch toward the workpiece.
        """
        if self.tool_z_points_into_workpiece:
            requested_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            requested_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        return self.make_tool_orientation(requested_axis)

    def joint_jump(
        self,
        q_new: np.ndarray,
        q_old: np.ndarray,
    ) -> float:
        """
        Returns the largest normalized joint-space jump.

        Revolute differences are wrapped to the shortest angular path.
        Prismatic motion is converted using 10 mm equivalent to 5 degrees;
        this is a planning heuristic, not a physical unit conversion.
        """
        q_new = self._validate_joint_vector(q_new)
        q_old = self._validate_joint_vector(q_old)

        maximum_jump_deg = 0.0

        for index, parameter in enumerate(self.dh_table):
            difference = float(q_new[index] - q_old[index])

            if parameter.joint_type == "revolute":
                difference = self._normalize_angle_difference(difference)
                jump_deg = float(np.rad2deg(abs(difference)))
            else:
                jump_deg = abs(difference) * 0.5

            maximum_jump_deg = max(maximum_jump_deg, jump_deg)

        return maximum_jump_deg

    def _normalized_joint_jump(
        self,
        q_new: np.ndarray,
        q_old: np.ndarray,
    ) -> float:
        """Backward-compatible alias for `joint_jump()`."""
        return self.joint_jump(q_new, q_old)

    @staticmethod
    def _rotation_to_quaternion(
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Converts a 3x3 rotation matrix to a quaternion [w, x, y, z]."""
        rotation = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(rotation))

        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            scalar = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = (
                np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            scalar = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = (
                np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            scalar = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = (
                np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            scalar = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

        return np.array([scalar, x, y, z], dtype=np.float64)

    @staticmethod
    def _rotation_error(
        current_rotation: np.ndarray,
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the SO(3) logarithm of target_rotation @ current_rotation.T.

        The result is a rotation vector in the base frame.
        """
        current_rotation = np.asarray(current_rotation, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)

        error_rotation = target_rotation @ current_rotation.T
        quaternion = RobotManipulator._rotation_to_quaternion(error_rotation)
        quaternion = quaternion / np.linalg.norm(quaternion)

        scalar = float(quaternion[0])
        vector = quaternion[1:]
        vector_norm = float(np.linalg.norm(vector))

        if vector_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)

        if scalar < 0.0:
            scalar = -scalar
            vector = -vector

        angle = 2.0 * float(np.arctan2(vector_norm, scalar))
        axis = vector / vector_norm

        return axis * angle

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iters: int = 500,
        pos_tol: float = 0.1,
        rot_tol: float = 0.015,
        position_weight: float = 1.0,
        orientation_weight: float = 20.0,
        initial_damping: float = 0.05,
        minimum_damping: float = 1e-4,
        maximum_damping: float = 5.0,
        max_step_norm: float = 0.35,
        max_joint_jump_deg: float | None = None,
    ) -> tuple[bool, np.ndarray, float]:
        """
        Solves N-DoF inverse kinematics using damped least squares.

        The solver minimizes a weighted position/orientation residual and uses
        backtracking to reject steps that increase the task error. When `q_init`
        is supplied, it prefers the valid solution with the smallest normalized
        joint-space jump from that initial configuration.

        Returns:
            A tuple of `(success, joint_positions, position_error_mm)`.
        """
        target_position = _as_vector3(target_pos, "target_pos")

        if target_rot is None:
            target_rotation = self.default_tool_orientation()
        else:
            target_rotation = np.asarray(target_rot, dtype=np.float64)

            if target_rotation.shape != (3, 3):
                raise ValueError("target_rot must be a 3x3 rotation matrix.")

            if not np.all(np.isfinite(target_rotation)):
                raise ValueError("target_rot must contain only finite values.")

        if max_iters < 1:
            raise ValueError("max_iters must be at least 1.")

        if pos_tol < 0.0 or rot_tol < 0.0:
            raise ValueError("Tolerances must be non-negative.")

        if position_weight <= 0.0 or orientation_weight < 0.0:
            raise ValueError(
                "position_weight must be positive and orientation_weight "
                "must be non-negative."
            )

        if not (minimum_damping <= initial_damping <= maximum_damping):
            raise ValueError(
                "Damping values must satisfy "
                "minimum_damping <= initial_damping <= maximum_damping."
            )

        if max_step_norm <= 0.0:
            raise ValueError("max_step_norm must be positive.")

        target_distance = float(np.linalg.norm(target_position))

        if target_distance > self.max_reach_mm:
            logger.warning(
                "[%s] IK rejected by coarse reach check: "
                "distance %.1f mm > max_reach %.1f mm.",
                self.robot_type,
                target_distance,
                self.max_reach_mm,
            )
            return (
                False,
                np.zeros(self.dof, dtype=np.float64),
                target_distance,
            )

        def evaluate(
            joints: np.ndarray,
        ) -> tuple[
            list[np.ndarray],
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
        ]:
            frames = self.forward_kinematics_chain(joints)
            pose = frames[-1]

            position_error = target_position - pose[:3, 3]
            position_error_norm = float(np.linalg.norm(position_error))

            if self.position_only_orientation:
                current_rotation = pose[:3, :3]

                current_yaw = float(
                    np.arctan2(
                        current_rotation[1, 0],
                        current_rotation[0, 0],
                    )
                )
                target_yaw = float(
                    np.arctan2(
                        target_rotation[1, 0],
                        target_rotation[0, 0],
                    )
                )

                yaw_error = float(
                    np.arctan2(
                        np.sin(target_yaw - current_yaw),
                        np.cos(target_yaw - current_yaw),
                    )
                )

                rotation_error = np.array(
                    [0.0, 0.0, yaw_error],
                    dtype=np.float64,
                )
            else:
                rotation_error = self._rotation_error(
                    pose[:3, :3],
                    target_rotation,
                )

            rotation_error_norm = float(np.linalg.norm(rotation_error))

            weighted_error = np.concatenate(
                [
                    position_weight * position_error,
                    orientation_weight * rotation_error,
                ]
            )

            task_cost = 0.5 * float(weighted_error @ weighted_error)

            return (
                frames,
                position_error,
                rotation_error,
                position_error_norm,
                rotation_error_norm,
                task_cost,
            )

        candidate_seeds = [seed.copy() for seed in self.seeds]

        if q_init is not None:
            initial_joints = self._validate_joint_vector(q_init)
            candidate_seeds.insert(0, initial_joints.copy())

        best_fallback_q = candidate_seeds[0].copy()
        best_fallback_position_error = float("inf")
        best_fallback_task_cost = float("inf")
        best_fallback_orientation_error = float("inf")

        best_valid_q: np.ndarray | None = None
        best_valid_joint_jump = float("inf")
        best_valid_position_error = float("inf")

        identity_6 = np.eye(6, dtype=np.float64)
        random_generator = np.random.default_rng(seed=42)

        for seed in candidate_seeds:
            q = self._clip_to_joint_limits(seed.copy())

            (
                frames,
                position_error,
                rotation_error,
                position_error_norm,
                rotation_error_norm,
                task_cost,
            ) = evaluate(q)

            damping = initial_damping
            stagnation_count = 0

            def record_solution(
                candidate_joints: np.ndarray,
                candidate_position_error: float,
                candidate_rotation_error: float,
                candidate_task_cost: float,
            ) -> bool:
                nonlocal best_valid_q
                nonlocal best_valid_joint_jump
                nonlocal best_valid_position_error
                nonlocal best_fallback_q
                nonlocal best_fallback_position_error
                nonlocal best_fallback_orientation_error
                nonlocal best_fallback_task_cost

                if candidate_position_error < best_fallback_position_error or (
                    candidate_position_error == best_fallback_position_error
                    and candidate_task_cost < best_fallback_task_cost
                ):
                    best_fallback_q = candidate_joints.copy()
                    best_fallback_position_error = candidate_position_error
                    best_fallback_orientation_error = candidate_rotation_error
                    best_fallback_task_cost = candidate_task_cost

                position_ok = candidate_position_error < pos_tol
                orientation_ok = candidate_rotation_error < rot_tol

                if not (position_ok and orientation_ok):
                    return False

                if q_init is None:
                    best_valid_q = candidate_joints.copy()
                    best_valid_position_error = candidate_position_error
                    return True

                joint_jump = self.joint_jump(
                    self._nearest_equivalent_configuration(
                        candidate_joints,
                        q_init,
                    ),
                    q_init,
                )

                if joint_jump < best_valid_joint_jump:
                    best_valid_q = self._nearest_equivalent_configuration(
                        candidate_joints,
                        q_init,
                    ).copy()
                    best_valid_joint_jump = joint_jump
                    best_valid_position_error = candidate_position_error

                return bool(
                    max_joint_jump_deg is not None and joint_jump <= max_joint_jump_deg
                )

            if record_solution(
                q,
                position_error_norm,
                rotation_error_norm,
                task_cost,
            ):

                assert best_valid_q is not None
                return True, best_valid_q.copy(), best_valid_position_error

            for _ in range(max_iters):
                if position_error_norm < pos_tol and rotation_error_norm < rot_tol:
                    if q_init is None:
                        return (
                            True,
                            q.copy(),
                            position_error_norm,
                        )

                    break

                weighted_error = np.concatenate(
                    [
                        position_weight * position_error,
                        orientation_weight * rotation_error,
                    ]
                )

                jacobian = self.compute_jacobian_from_frames(frames)

                try:
                    damped_matrix = jacobian @ jacobian.T + damping**2 * identity_6
                    full_step = jacobian.T @ np.linalg.solve(
                        damped_matrix,
                        weighted_error,
                    )
                except np.linalg.LinAlgError:
                    damping = min(
                        maximum_damping,
                        damping * 2.0,
                    )
                    continue

                if not np.all(np.isfinite(full_step)):
                    damping = min(
                        maximum_damping,
                        damping * 2.0,
                    )
                    continue

                step_norm = float(np.linalg.norm(full_step))

                if step_norm > max_step_norm:
                    full_step *= max_step_norm / step_norm

                accepted = False

                for step_scale in (1.0, 0.5, 0.25, 0.1):
                    candidate_q = self._clip_to_joint_limits(q + step_scale * full_step)

                    (
                        candidate_frames,
                        candidate_position_error,
                        candidate_rotation_error,
                        candidate_position_error_norm,
                        candidate_rotation_error_norm,
                        candidate_task_cost,
                    ) = evaluate(candidate_q)

                    if candidate_task_cost < task_cost:
                        q = candidate_q
                        frames = candidate_frames
                        position_error = candidate_position_error
                        rotation_error = candidate_rotation_error
                        position_error_norm = candidate_position_error_norm
                        rotation_error_norm = candidate_rotation_error_norm
                        task_cost = candidate_task_cost

                        damping = max(
                            minimum_damping,
                            damping * 0.5,
                        )
                        stagnation_count = 0
                        accepted = True

                        if record_solution(
                            q,
                            position_error_norm,
                            rotation_error_norm,
                            task_cost,
                        ):

                            assert best_valid_q is not None
                            return True, best_valid_q.copy(), best_valid_position_error

                        break

                if not accepted:
                    damping = min(
                        maximum_damping,
                        damping * 2.0,
                    )
                    stagnation_count += 1

                    if damping >= maximum_damping and stagnation_count >= 5:
                        perturbation = random_generator.normal(
                            0.0,
                            0.01,
                            size=self.dof,
                        )
                        q = self._clip_to_joint_limits(q + perturbation)

                        (
                            frames,
                            position_error,
                            rotation_error,
                            position_error_norm,
                            rotation_error_norm,
                            task_cost,
                        ) = evaluate(q)

                        damping = initial_damping
                        stagnation_count = 0

                        if record_solution(
                            q,
                            position_error_norm,
                            rotation_error_norm,
                            task_cost,
                        ):

                            return (
                                True,
                                q.copy(),
                                position_error_norm,
                            )

        if best_valid_q is not None:
            return (
                True,
                best_valid_q.copy(),
                best_valid_position_error,
            )
        logger.warning(
            "[%s] IK failed for target %s: position_error=%.6f mm, "
            "orientation_error=%.6f rad (%.3f deg).",
            self.robot_type,
            np.round(target_position, 3).tolist(),
            best_fallback_position_error,
            best_fallback_orientation_error,
            np.rad2deg(best_fallback_orientation_error),
        )
        return (
            False,
            best_fallback_q.copy(),
            best_fallback_position_error,
        )

    @staticmethod
    def _normalize_angle_difference(
        angle_difference: float,
    ) -> float:
        """Wraps an angular difference into [-pi, pi]."""
        return (angle_difference + np.pi) % (2.0 * np.pi) - np.pi

    def _nearest_equivalent_configuration(
        self,
        q_candidate: np.ndarray,
        q_reference: np.ndarray,
    ) -> np.ndarray:
        """
        Shifts revolute joints by whole turns to the equivalent configuration
        nearest to q_reference, while respecting configured joint limits.
        """
        q_candidate = self._validate_joint_vector(q_candidate)
        q_reference = self._validate_joint_vector(q_reference)

        adjusted = q_candidate.copy()

        for index, parameter in enumerate(self.dh_table):
            if parameter.joint_type != "revolute":
                continue

            lower, upper = self.joint_limits[index]
            raw_difference = float(q_candidate[index] - q_reference[index])
            wrapped_difference = self._normalize_angle_difference(raw_difference)
            nearest_candidate = float(q_reference[index]) + wrapped_difference

            if lower <= nearest_candidate <= upper:
                adjusted[index] = nearest_candidate

        return self._clip_to_joint_limits(adjusted)


SUPPORTED_ROBOTS: dict[str, str] = {
    "industrial_6dof": (
        "6-DoF industrial welding arm with spherical wrist "
        "(KUKA/Mitsubishi-style kinematic profile)."
    ),
    "cobot_6dof": (
        "6-DoF collaborative welding arm with offset wrist "
        "(UR5/UR10-inspired synthetic kinematic profile)."
    ),
    "scara_4dof": (
        "4-DoF SCARA vertical spot-welding robot " "(R-R-P-R kinematic chain)."
    ),
}


def create_manipulator(
    robot_type: str = "industrial_6dof",
    config_path: Path | str | None = None,
    config_dir: Path | str | None = None,
) -> RobotManipulator:
    """
    Creates a configured kinematic model from a JSON config file or
    registered robot type.
    """
    if config_path is not None:
        config = load_robot_config_file(config_path)
        return _build_manipulator_from_config(config)

    registry = _get_robot_registry(config_dir)
    key = robot_type.strip().lower()

    if key not in registry:
        raise ValueError(
            f"Unknown robot_type '{robot_type}'. Supported models: "
            f"{sorted(registry.keys())}."
        )

    return _build_manipulator_from_config(registry[key])


class SixDofArm(RobotManipulator):
    """Backward-compatible default 6-DoF industrial manipulator."""

    def __init__(self) -> None:
        default_arm = create_manipulator("industrial_6dof")

        super().__init__(
            robot_type=default_arm.robot_type,
            description=default_arm.description,
            dh_table=default_arm.dh_table,
            joint_limits=default_arm.joint_limits,
            seeds=default_arm.seeds,
            max_reach_mm=default_arm.max_reach_mm,
            position_only_orientation=(default_arm.position_only_orientation),
            tool_z_points_into_workpiece=(default_arm.tool_z_points_into_workpiece),
        )


def _as_vector3(value: object, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{field_name} must contain exactly three coordinates.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain only finite values.")

    return vector
