from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal
import numpy as np

logger = logging.getLogger("mcp_robot_welding.kinematics")

JointType = Literal["revolute", "prismatic"]
OrientationMode = Literal["full_so3", "yaw_only"]


@dataclass(frozen=True)
class DHParameter:
    """
    Standard Denavit-Hartenberg (DH) parameters for a single manipulator link.

    Units:
        - a: link length in millimeters (mm)
        - alpha: link twist in radians (rad)
        - d: link offset along Z in millimeters (mm)
        - theta_offset: home joint angle offset in radians (rad)
    """

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0
    joint_type: JointType = "revolute"


class RobotManipulator:
    """Generic N-DoF industrial manipulator supporting revolute and prismatic joints."""

    def __init__(
        self,
        robot_type: str,
        description: str,
        dh_table: list[DHParameter],
        joint_limits: list[tuple[float, float]],
        seeds: list[np.ndarray],
        max_reach_mm: float,
        orientation_mode: OrientationMode = "full_so3",
        prismatic_mm_per_equiv_deg: float = 2.0,
    ) -> None:
        if len(dh_table) != len(joint_limits):
            raise ValueError("dh_table and joint_limits must have equal length.")

        self.robot_type = robot_type
        self.description = description
        self.dh_table = dh_table
        self.joint_limits = joint_limits
        self.seeds = seeds
        self.max_reach_mm = max_reach_mm
        self.orientation_mode = orientation_mode
        self.prismatic_mm_per_equiv_deg = prismatic_mm_per_equiv_deg
        self.dof = len(dh_table)

    def _validate_joints(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64)
        if q.shape != (self.dof,):
            raise ValueError(
                f"[{self.robot_type}] Expected joints shape ({self.dof},), got {q.shape}."
            )
        return q

    @staticmethod
    def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
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

    def forward_kinematics_chain(self, joints: np.ndarray) -> list[np.ndarray]:
        """Returns homogeneous 4x4 SE(3) matrices for base and every joint frame."""
        q = self._validate_joints(joints)
        frames: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        t_current = np.eye(4, dtype=np.float64)

        for idx, param in enumerate(self.dh_table):
            q_val = float(q[idx])
            if param.joint_type == "prismatic":
                theta = param.theta_offset
                d = param.d + q_val
            else:
                theta = q_val + param.theta_offset
                d = param.d

            t_link = self._dh_transform(param.a, param.alpha, d, theta)
            t_current = t_current @ t_link
            frames.append(t_current.copy())

        return frames

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Returns 4x4 SE(3) pose of the welding torch TCP."""
        return self.forward_kinematics_chain(joints)[-1]

    def compute_jacobian_from_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Computes the 6xN geometric Jacobian from precomputed FK frames.
        - Revolute joint:  J_v = z_i x (p_tcp - p_i),  J_w = z_i
        - Prismatic joint: J_v = z_i,                  J_w = 0
        """
        tcp_position = frames[-1][:3, 3]
        jacobian = np.zeros((6, self.dof), dtype=np.float64)

        for idx, param in enumerate(self.dh_table):
            joint_frame = frames[idx]
            joint_axis = joint_frame[:3, 2]
            joint_origin = joint_frame[:3, 3]

            if param.joint_type == "prismatic":
                jacobian[:3, idx] = joint_axis
                jacobian[3:, idx] = 0.0
            else:
                jacobian[:3, idx] = np.cross(joint_axis, tcp_position - joint_origin)
                jacobian[3:, idx] = joint_axis

        return jacobian

    def compute_jacobian(self, joints: np.ndarray) -> np.ndarray:
        """Computes the 6xN geometric Jacobian in the base frame."""
        frames = self.forward_kinematics_chain(joints)
        return self.compute_jacobian_from_frames(frames)

    @staticmethod
    def _rotation_error(
        current_rotation: np.ndarray,
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Computes numerically stable 3D rotation-vector error on SO(3),
        including explicit handling near angle = pi where sin(angle) -> 0.
        """
        r_err = target_rotation @ current_rotation.T
        cos_angle = float(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0))
        angle = float(np.arccos(cos_angle))

        if angle < 1e-8:
            return np.zeros(3, dtype=np.float64)

        # Near pi, extract rotation axis from symmetric part (R + I) / 2
        if np.pi - angle < 1e-4:
            diag = np.maximum(0.0, (np.diag(r_err) + 1.0) / 2.0)
            axis = np.sqrt(diag)
            if r_err.item(2, 1) - r_err.item(1, 2) < 0:
                axis[0] = -float(axis.item(0))
            if r_err.item(0, 2) - r_err.item(2, 0) < 0:
                axis[1] = -float(axis.item(1))
            if r_err.item(1, 0) - r_err.item(0, 1) < 0:
                axis[2] = -float(axis.item(2))
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                return (axis / norm) * angle

        axis = np.array(
            [
                r_err.item(2, 1) - r_err.item(1, 2),
                r_err.item(0, 2) - r_err.item(2, 0),
                r_err.item(1, 0) - r_err.item(0, 1),
            ],
            dtype=np.float64,
        ) / (2.0 * np.sin(angle))

        return axis * angle


    def _clip_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        clipped = joints.copy()
        for index, (lower, upper) in enumerate(self.joint_limits):
            value = float(clipped[index])
            if self.dh_table[index].joint_type == "revolute":
                if (upper - lower) >= 2.0 * np.pi:
                    value = (value + np.pi) % (2.0 * np.pi) - np.pi
            clipped[index] = np.clip(value, lower, upper)
        return clipped

    def make_tool_orientation(self, tool_z_axis: np.ndarray) -> np.ndarray:
        """Builds a right-handed SO(3) orientation matrix with the requested tool Z axis."""
        tool_z_axis = np.asarray(tool_z_axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(tool_z_axis))
        if norm < 1e-9:
            raise ValueError("tool_z_axis must be a non-zero 3D vector.")
        tool_z_axis = tool_z_axis / norm

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper, tool_z_axis))) > 0.99:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        tool_x_axis = np.cross(helper, tool_z_axis)
        tool_x_axis = tool_x_axis / np.linalg.norm(tool_x_axis)
        tool_y_axis = np.cross(tool_z_axis, tool_x_axis)

        return np.column_stack([tool_x_axis, tool_y_axis, tool_z_axis])

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iters: int = 500,
        pos_tol: float = 0.1,
        rot_tol: float = 0.015,
        max_allowed_joint_jump_deg: float = 45.0,
    ) -> tuple[bool, np.ndarray, float]:
        """
        Solves N-DoF IK using unit-scaled, adaptive Damped Least Squares
        in 6D task space: delta_q = J^T (J J^T + lambda^2 I)^-1 e.
        """
        target_position = np.asarray(target_pos, dtype=np.float64)
        if target_position.shape != (3,):
            raise ValueError(f"Expected target_pos shape (3,), got {target_position.shape}.")

        target_distance = float(np.linalg.norm(target_position))
        if target_distance > self.max_reach_mm:
            logger.warning(
                "[%s] IK reach heuristic rejected: %.1f mm > %.1f mm",
                self.robot_type,
                target_distance,
                self.max_reach_mm,
            )
            return False, np.zeros(self.dof, dtype=np.float64), target_distance

        if target_rot is None:
            target_rotation = self.make_tool_orientation(
                np.array([0.0, 0.0, -1.0], dtype=np.float64)
            )
        else:
            target_rotation = np.asarray(target_rot, dtype=np.float64)
            if target_rotation.shape != (3, 3):
                raise ValueError(f"Expected target_rot shape (3, 3), got {target_rotation.shape}.")

        validated_q_init = self._validate_joints(q_init) if q_init is not None else None
        candidate_seeds = [seed.copy() for seed in self.seeds]
        if validated_q_init is not None:
            candidate_seeds.insert(0, validated_q_init.copy())

        best_q = next(iter(candidate_seeds)).copy()
        best_position_error = float("inf")

        best_smooth_q: np.ndarray | None = None
        best_smooth_joint_jump = float("inf")
        best_smooth_position_error = float("inf")

        # Explicit unit scaling: 1 rad orientation error ~= 50 mm position error
        position_weight = 1.0
        orientation_weight = 50.0

        for seed in candidate_seeds:
            q = self._clip_to_joint_limits(seed.copy())
            position_error_norm = float("inf")

            for _ in range(max_iters):
                frames = self.forward_kinematics_chain(q)
                current_pose = frames[-1]
                current_position = current_pose[:3, 3]
                current_rotation = current_pose[:3, :3]
                position_error = target_position - current_position
                position_error_norm = float(np.linalg.norm(position_error))

                if self.orientation_mode == "yaw_only":
                    yaw_cur = float(
                        np.arctan2(
                            current_rotation.item(1, 0),
                            current_rotation.item(0, 0),
                        )
                    )
                    yaw_tgt = float(
                        np.arctan2(
                            target_rotation.item(1, 0),
                            target_rotation.item(0, 0),
                        )
                    )
                    yaw_err = float(
                        np.arctan2(
                            np.sin(yaw_tgt - yaw_cur), np.cos(yaw_tgt - yaw_cur)
                        )
                    )
                    orientation_error = np.array(
                        [0.0, 0.0, yaw_err], dtype=np.float64
                    )

                else:
                    orientation_error = self._rotation_error(
                        current_rotation, target_rotation
                    )

                orientation_error_norm = float(np.linalg.norm(orientation_error))

                if position_error_norm < best_position_error:
                    best_position_error = position_error_norm
                    best_q = q.copy()

                if (
                    position_error_norm < pos_tol
                    and orientation_error_norm < rot_tol
                ):
                    if validated_q_init is None:
                        return True, q, position_error_norm

                    joint_jump = self._normalized_joint_jump(
                        q, validated_q_init
                    )
                    if joint_jump < best_smooth_joint_jump:
                        best_smooth_q = q.copy()
                        best_smooth_joint_jump = joint_jump
                        best_smooth_position_error = position_error_norm

                    # Early exit when a smooth continuous solution is already found
                    if best_smooth_joint_jump <= max_allowed_joint_jump_deg:
                        return True, best_smooth_q, best_smooth_position_error

                w_diag = np.array([1.0, 1.0, 1.0, 50.0, 50.0, 50.0], dtype=np.float64)
                weighted_error = w_diag * np.concatenate(
                    [position_error, orientation_error]
                )
                jacobian = self.compute_jacobian_from_frames(frames)
                weighted_jac = w_diag[:, None] * jacobian

                cond = float(np.linalg.cond(weighted_jac))
                damping = 0.05 if cond > 1e4 else 0.01
                lambda_sq = damping**2

                jjt = weighted_jac @ weighted_jac.T + lambda_sq * np.eye(6)
                delta_q = weighted_jac.T @ np.linalg.solve(jjt, weighted_error)


                # Scale full step vector in normalized units to preserve descent direction
                scaled_dq = np.array(
                    [
                        (delta_q[i] / 50.0)
                        if self.dh_table[i].joint_type == "prismatic"
                        else delta_q[i]
                        for i in range(self.dof)
                    ],
                    dtype=np.float64,
                )
                step_norm = float(np.linalg.norm(scaled_dq))
                max_step_norm = 0.35
                if step_norm > max_step_norm:
                    delta_q *= max_step_norm / step_norm

                q = self._clip_to_joint_limits(q + delta_q)

        if best_smooth_q is not None:
            return True, best_smooth_q, best_smooth_position_error

        return best_position_error <= pos_tol, best_q, best_position_error

    def _normalized_joint_jump(
        self, q_new: np.ndarray, q_old: np.ndarray
    ) -> float:
        """Computes max joint jump in equivalent degrees using configurable prismatic scaling."""
        max_jump_deg = 0.0
        for idx, param in enumerate(self.dh_table):
            diff = abs(float(q_new[idx]) - float(q_old[idx]))
            if param.joint_type == "prismatic":
                jump_deg = diff / max(1e-6, self.prismatic_mm_per_equiv_deg)
            else:
                jump_deg = float(np.rad2deg(diff))
            max_jump_deg = max(max_jump_deg, jump_deg)
        return max_jump_deg


SUPPORTED_ROBOTS: dict[str, str] = {
    "industrial_6dof": "6-DoF Industrial Welding Arm with Spherical Wrist (KUKA/Mitsubishi style)",
    "cobot_6dof": "6-DoF Collaborative Welding Arm with Synthetic Offset Wrist (UR-inspired)",
    "scara_4dof": "4-DoF SCARA Vertical Spot-Welding Robot (R-R-P-R kinematic chain)",
}


def create_manipulator(robot_type: str = "industrial_6dof") -> RobotManipulator:
    """Factory creating configured kinematic models for supported industrial robots."""
    key = robot_type.strip().lower()

    if key == "industrial_6dof":
        return RobotManipulator(
            robot_type="industrial_6dof",
            description=SUPPORTED_ROBOTS["industrial_6dof"],
            dh_table=[
                DHParameter(a=150.0, alpha=-np.pi / 2, d=400.0),
                DHParameter(a=550.0, alpha=0.0, d=0.0, theta_offset=-np.pi / 2),
                DHParameter(a=120.0, alpha=-np.pi / 2, d=0.0),
                DHParameter(a=0.0, alpha=np.pi / 2, d=500.0),
                DHParameter(a=0.0, alpha=-np.pi / 2, d=0.0),
                DHParameter(a=0.0, alpha=0.0, d=120.0),
            ],
            joint_limits=[
                (-np.deg2rad(170), np.deg2rad(170)),
                (-np.deg2rad(135), np.deg2rad(135)),
                (-np.deg2rad(150), np.deg2rad(160)),
                (-np.deg2rad(185), np.deg2rad(185)),
                (-np.deg2rad(135), np.deg2rad(135)),
                (-np.deg2rad(350), np.deg2rad(350)),
            ],
            seeds=[
                np.zeros(6, dtype=np.float64),
                np.array([0.0, 0.2, 0.5, 0.0, 0.8, 0.0], dtype=np.float64),
                np.array([0.0, -0.3, 0.8, 0.0, 0.9, 0.0], dtype=np.float64),
            ],
            max_reach_mm=1350.0,
        )

    if key == "cobot_6dof":
        # Synthetic non-spherical offset wrist geometry (reach ~1150 mm)
        return RobotManipulator(
            robot_type="cobot_6dof",
            description=SUPPORTED_ROBOTS["cobot_6dof"],
            dh_table=[
                DHParameter(a=0.0, alpha=np.pi / 2, d=180.0),
                DHParameter(a=-500.0, alpha=0.0, d=0.0),
                DHParameter(a=-450.0, alpha=0.0, d=0.0),
                DHParameter(a=0.0, alpha=np.pi / 2, d=130.0),
                DHParameter(a=0.0, alpha=-np.pi / 2, d=100.0),
                DHParameter(a=0.0, alpha=0.0, d=110.0),
            ],
            joint_limits=[
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.deg2rad(165), np.deg2rad(165)),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-2.0 * np.pi, 2.0 * np.pi),
            ],
            seeds=[
                np.array([2.9, -2.1, 1.5, -1.0, -1.57, 0.0], dtype=np.float64),
                np.array([-2.9, -2.1, 1.5, -1.0, -1.57, 0.0], dtype=np.float64),
                np.array([3.14, -1.8, 1.2, -1.0, -1.57, 0.0], dtype=np.float64),
            ],
            max_reach_mm=1200.0,
        )

    if key == "scara_4dof":
        # 4-DoF R-R-P-R SCARA: horizontal reach 450 + 400 = 850 mm, vertical Z quill [0..350 mm]
        return RobotManipulator(
            robot_type="scara_4dof",
            description=SUPPORTED_ROBOTS["scara_4dof"],
            dh_table=[
                DHParameter(a=450.0, alpha=0.0, d=520.0, joint_type="revolute"),
                DHParameter(a=400.0, alpha=np.pi, d=0.0, joint_type="revolute"),
                DHParameter(a=0.0, alpha=0.0, d=0.0, joint_type="prismatic"),
                DHParameter(a=0.0, alpha=0.0, d=60.0, joint_type="revolute"),
            ],
            joint_limits=[
                (-np.deg2rad(150), np.deg2rad(150)),
                (-np.deg2rad(145), np.deg2rad(145)),
                (0.0, 350.0),  # Prismatic vertical stroke in mm (downward along flipped Z)
                (-np.pi, np.pi),
            ],
            seeds=[
                np.array([0.4, -0.8, 100.0, 0.0], dtype=np.float64),
                np.array([-0.4, 0.8, 100.0, 0.0], dtype=np.float64),
            ],
            max_reach_mm=1100.0,
            orientation_mode="yaw_only",
        )

    raise ValueError(
        f"Unknown robot_type '{robot_type}'. Supported models: {sorted(SUPPORTED_ROBOTS.keys())}."
    )


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
        )
