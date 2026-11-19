from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("mcp_robot_welding.kinematics")


@dataclass(frozen=True)
class DHParameter:
    a: float      # Link length (mm)
    alpha: float  # Link twist (rad)
    d: float      # Link offset (mm)
    theta_offset: float = 0.0


class SixDofArm:
    """Kinematic model of a 6-DoF industrial welding manipulator (KUKA/Mitsubishi style)."""

    def __init__(self) -> None:
        self.dh_table: list[DHParameter] = [
            DHParameter(a=150.0, alpha=-np.pi / 2, d=400.0),
            DHParameter(a=550.0, alpha=0.0, d=0.0, theta_offset=-np.pi / 2),
            DHParameter(a=120.0, alpha=-np.pi / 2, d=0.0),
            DHParameter(a=0.0, alpha=np.pi / 2, d=500.0),
            DHParameter(a=0.0, alpha=-np.pi / 2, d=0.0),
            DHParameter(a=0.0, alpha=0.0, d=120.0),  # Welding torch TCP offset
        ]
        self.joint_limits: list[tuple[float, float]] = [
            (-np.deg2rad(170), np.deg2rad(170)),
            (-np.deg2rad(135), np.deg2rad(135)),
            (-np.deg2rad(150), np.deg2rad(160)),
            (-np.deg2rad(185), np.deg2rad(185)),
            (-np.deg2rad(135), np.deg2rad(135)),
            (-np.deg2rad(350), np.deg2rad(350)),
        ]
        self.max_reach_mm: float = 1350.0

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
        """Returns homogeneous 4x4 matrices for base and every joint frame."""
        frames: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        t_current = np.eye(4, dtype=np.float64)

        for idx, param in enumerate(self.dh_table):
            theta = float(joints.item(idx)) + param.theta_offset
            t_link = self._dh_transform(
                param.a,
                param.alpha,
                param.d,
                theta,
            )
            t_current = t_current @ t_link
            frames.append(t_current.copy())

        return frames

    def compute_jacobian(self, joints: np.ndarray) -> np.ndarray:
        """
        Computes the 6x6 geometric Jacobian.

        Linear velocity is expressed in the base frame.
        Angular velocity is expressed in the base frame.

        For a revolute joint:
            J_v = z_i x (p_tcp - p_i)
            J_w = z_i
        """
        frames = self.forward_kinematics_chain(joints)
        tcp_position = frames[-1][:3, 3]

        jacobian = np.zeros((6, 6), dtype=np.float64)

        for joint_index in range(6):
            joint_frame = frames[joint_index]

            joint_axis = joint_frame[:3, 2]
            joint_origin = joint_frame[:3, 3]

            jacobian[:3, joint_index] = np.cross(
                joint_axis,
                tcp_position - joint_origin,
            )
            jacobian[3:, joint_index] = joint_axis

        return jacobian
    
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Returns 4x4 SE(3) pose of the welding torch TCP."""
        return self.forward_kinematics_chain(joints)[-1]
    
    @staticmethod
    def _rotation_error(
        current_rotation: np.ndarray,
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the rotation-vector error from current to target orientation.

        The result is a 3D vector in the base frame.
        """
        rotation_error = target_rotation @ current_rotation.T

        angle = np.arccos(
            np.clip(
                (np.trace(rotation_error) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )

        if angle < 1e-8:
            return np.zeros(3, dtype=np.float64)

        axis = np.array(
            [
                rotation_error[2, 1] - rotation_error[1, 2],
                rotation_error[0, 2] - rotation_error[2, 0],
                rotation_error[1, 0] - rotation_error[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * np.sin(angle))

        return axis * angle
    
    def _clip_to_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        clipped = joints.copy()

        for index, (lower, upper) in enumerate(self.joint_limits):
            clipped[index] = np.clip(
                float(clipped[index]),
                lower,
                upper,
            )

        return clipped
    
    def make_tool_orientation(self, tool_z_axis: np.ndarray) -> np.ndarray:
        """
        Builds a right-handed orientation matrix with the requested tool Z axis.
        """
        tool_z_axis = np.asarray(tool_z_axis, dtype=np.float64)
        tool_z_axis = tool_z_axis / np.linalg.norm(tool_z_axis)

        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if abs(float(np.dot(helper, tool_z_axis))) > 0.99:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        tool_x_axis = np.cross(helper, tool_z_axis)
        tool_x_axis = tool_x_axis / np.linalg.norm(tool_x_axis)

        tool_y_axis = np.cross(tool_z_axis, tool_x_axis)

        return np.column_stack(
            [
                tool_x_axis,
                tool_y_axis,
                tool_z_axis,
            ]
        )
        
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iters: int = 500,
        pos_tol: float = 0.1,
        rot_tol: float = 0.01,
    ) -> tuple[bool, np.ndarray, float]:
        """
        Solves 6-DoF welding torch IK using damped least squares.

        When an initial joint configuration is provided, the solver searches
        multiple seeds and selects the feasible solution with the smallest
        joint-space jump relative to the initial configuration.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_distance = float(np.linalg.norm(target_pos))

        if target_distance > self.max_reach_mm:
            logger.warning(
                "IK rejected: target distance %.1f mm exceeds max reach %.1f mm",
                target_distance,
                self.max_reach_mm,
            )

            return False, np.zeros(6), target_distance

        if target_rot is None:
            target_rotation = self.make_tool_orientation(
                np.array([0.0, 0.0, -1.0], dtype=np.float64)
            )

        else:
            target_rotation = np.asarray(
                target_rot,
                dtype=np.float64,
            )

        seeds = [
            np.zeros(6, dtype=np.float64),
            np.array(
                [0.0, 0.2, 0.5, 0.0, 0.8, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [0.0, -0.3, 0.8, 0.0, 0.9, 0.0],
                dtype=np.float64,
            ),
        ]

        if q_init is not None:
            seeds.insert(
                0,
                np.asarray(q_init, dtype=np.float64).copy(),
            )

        best_q = seeds[0].copy()
        best_position_error = float("inf")

        best_smooth_q: np.ndarray | None = None
        best_smooth_joint_jump = float("inf")
        best_smooth_position_error = float("inf")

        for seed_index, seed in enumerate(seeds):
            q = self._clip_to_joint_limits(seed.copy())

            position_error_norm = float("inf")
            orientation_error_norm = float("inf")

            for iteration in range(max_iters):
                current_pose = self.forward_kinematics(q)

                current_position = current_pose[:3, 3]
                current_rotation = current_pose[:3, :3]

                position_error = target_pos - current_position

                orientation_error = self._rotation_error(
                    current_rotation,
                    target_rotation,
                )

                position_error_norm = float(
                    np.linalg.norm(position_error)
                )

                orientation_error_norm = float(
                    np.linalg.norm(orientation_error)
                )

                if position_error_norm < best_position_error:
                    best_position_error = position_error_norm
                    best_q = q.copy()

                if (
                    position_error_norm < pos_tol
                    and orientation_error_norm < rot_tol
                ):
                    logger.debug(
                        "IK candidate found: seed=%d iteration=%d "
                        "position_error=%.4f mm "
                        "orientation_error=%.6f",
                        seed_index,
                        iteration,
                        position_error_norm,
                        orientation_error_norm,
                    )

                    if q_init is None:
                        return (
                            True,
                            q,
                            position_error_norm,
                        )

                    joint_jump = float(
                        np.max(np.abs(q - q_init))
                    )

                    if joint_jump < best_smooth_joint_jump:
                        best_smooth_q = q.copy()
                        best_smooth_joint_jump = joint_jump
                        best_smooth_position_error = (
                            position_error_norm
                        )

                error_vector = np.concatenate(
                    [
                        position_error,
                        orientation_error,
                    ]
                )

                jacobian = self.compute_jacobian(q)

                damping = 1e-4

                delta_q = np.linalg.solve(
                    jacobian.T @ jacobian
                    + damping * np.eye(6),
                    jacobian.T @ error_vector,
                )

                max_delta = float(np.max(np.abs(delta_q)))

                if max_delta > 0.3:
                    delta_q *= 0.3 / max_delta

                q = self._clip_to_joint_limits(q + delta_q)

            logger.debug(
                "IK seed finished: seed=%d "
                "final_position_error=%.2f mm "
                "final_orientation_error=%.6f",
                seed_index,
                position_error_norm,
                orientation_error_norm,
            )

        if best_smooth_q is not None:
            logger.debug(
                "IK selected smoothest solution: "
                "joint_jump=%.2f degrees "
                "position_error=%.4f mm",
                np.rad2deg(best_smooth_joint_jump),
                best_smooth_position_error,
            )

            return (
                True,
                best_smooth_q,
                best_smooth_position_error,
            )

        logger.warning(
            "IK failed across all seeds: "
            "best_position_error=%.2f mm",
            best_position_error,
        )

        return (
            best_position_error <= pos_tol,
            best_q,
            best_position_error,
        )