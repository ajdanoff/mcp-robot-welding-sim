from __future__ import annotations


from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, field_validator


class WeldSeam(BaseModel):
    seam_id: str = Field(
        ...,
        description="Unique identifier of the weld seam",
    )

    description: str = Field(
        ...,
        description="Human-readable seam description",
    )

    start_xyz: list[float] = Field(
        ...,
        description="[x, y, z] start point in mm",
    )

    end_xyz: list[float] = Field(
        ...,
        description="[x, y, z] end point in mm",
    )

    normal_xyz: list[float] = Field(
        default=[0.0, 0.0, 1.0],
        description="Surface normal vector for torch approach",
    )

    @field_validator("start_xyz", "end_xyz", "normal_xyz")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        vector = np.asarray(value, dtype=np.float64)

        if vector.shape != (3,):
            raise ValueError(
                "Vector must contain exactly three coordinates."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Vector must contain only finite values."
            )

        return value

    @field_validator("normal_xyz")
    @classmethod
    def validate_normal(cls, value: list[float]) -> list[float]:
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))

        if norm < 1e-9:
            raise ValueError(
                "Normal vector must be non-zero."
            )

        return value


@dataclass(frozen=True)
class AABB:
    name: str
    min_pt: np.ndarray
    max_pt: np.ndarray
    collision_margin_mm: float = 5.0

    def __post_init__(self) -> None:
        min_pt = np.asarray(self.min_pt, dtype=np.float64)
        max_pt = np.asarray(self.max_pt, dtype=np.float64)

        if min_pt.shape != (3,) or max_pt.shape != (3,):
            raise ValueError(
                "Bounding box points must contain exactly "
                "three coordinates."
            )

        if not np.all(np.isfinite(min_pt)) or not np.all(
            np.isfinite(max_pt)
        ):
            raise ValueError(
                "Bounding box points must contain only "
                "finite values."
            )

        if np.any(min_pt > max_pt):
            raise ValueError(
                "Bounding box minimum point must not exceed "
                "maximum point."
            )

        object.__setattr__(self, "min_pt", min_pt)
        object.__setattr__(self, "max_pt", max_pt)

    def contains_point(
        self,
        point: np.ndarray,
        margin: float = 0.0,
    ) -> bool:
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (3,):
            raise ValueError(
                "Point must contain exactly three coordinates."
            )

        lower_bound = self.min_pt - margin
        upper_bound = self.max_pt + margin

        return bool(
            np.all(point >= lower_bound)
            and np.all(point <= upper_bound)
        )


class WorkpieceScene:
    """
    Represents the fixture table, CAD workpiece bounding volume,
    and target weld seams.
    """

    def __init__(self) -> None:
        self.part_name = "Bracket_Assembly_REV_B"

        self.obstacles: list[AABB] = [
            AABB(
                name="Welding_Table",
                min_pt=np.array(
                    [-1000.0, -1000.0, -200.0],
                    dtype=np.float64,
                ),
                max_pt=np.array(
                    [1500.0, 1000.0, -1.0],
                    dtype=np.float64,
                ),
                collision_margin_mm=0.0,
            ),
            AABB(
                name="Workpiece_Core_Body",
                min_pt=np.array(
                    [460.0, -140.0, 0.0],
                    dtype=np.float64,
                ),
                max_pt=np.array(
                    [740.0, 140.0, 175.0],
                    dtype=np.float64,
                ),
                collision_margin_mm=5.0,
            ),
        ]

        self.seams: dict[str, WeldSeam] = {
            "seam_top_rib": WeldSeam(
                seam_id="seam_top_rib",
                description=(
                    "Top longitudinal rib weld along the upper plate"
                ),
                start_xyz=[480.0, 0.0, 180.0],
                end_xyz=[720.0, 0.0, 180.0],
                normal_xyz=[0.0, 0.0, 1.0],
            ),
            "seam_left_flange": WeldSeam(
                seam_id="seam_left_flange",
                description="Left side flange fillet weld",
                start_xyz=[500.0, -145.0, 90.0],
                end_xyz=[700.0, -145.0, 90.0],
                normal_xyz=[0.0, -0.707, 0.707],
            ),
            "seam_right_flange": WeldSeam(
                seam_id="seam_right_flange",
                description="Right side flange fillet weld",
                start_xyz=[500.0, 145.0, 90.0],
                end_xyz=[700.0, 145.0, 90.0],
                normal_xyz=[0.0, 0.707, 0.707],
            ),
        }

    def check_chain_collision(
        self,
        joint_frames: list[np.ndarray],
        collision_margin_mm: float = 5.0,
        samples_per_link: int = 8,
    ) -> tuple[bool, str]:
        """
        Checks whether robot links intersect scene obstacles.

        The base-to-first-joint segment is excluded because the robot
        base is assumed to be mounted outside the collision volume.
        """
        if not joint_frames:
            return False, "No joint frames provided."

        if len(joint_frames) < 3:
            return False, (
                "Not enough joint frames for collision checking."
            )

        for frame in joint_frames:
            frame_array = np.asarray(frame, dtype=np.float64)

            if frame_array.shape != (4, 4):
                return False, (
                    "Invalid joint frame: expected a 4x4 matrix."
                )

            if not np.all(np.isfinite(frame_array)):
                return False, (
                    "Invalid joint frame: contains non-finite values."
                )

        if samples_per_link < 2:
            raise ValueError(
                "samples_per_link must be at least 2."
            )

        last_frame_index = len(joint_frames) - 1

        for link_index in range(2, len(joint_frames)):
            previous_point = joint_frames[link_index - 1][:3, 3]
            current_point = joint_frames[link_index][:3, 3]

            sample_parameters = np.linspace(
                0.0,
                1.0,
                samples_per_link,
            )

            for sample_parameter in sample_parameters:
                sample_point = (
                    previous_point * (1.0 - sample_parameter)
                    + current_point * sample_parameter
                )

                is_tcp_tip = (
                    link_index == last_frame_index
                    and sample_parameter == 1.0
                )

                for obstacle in self.obstacles:
                    margin = (
                        0.0
                        if is_tcp_tip
                        else obstacle.collision_margin_mm
                    )

                    if obstacle.contains_point(
                        sample_point,
                        margin=margin,
                    ):
                        return (
                            True,
                            (
                                f"Link_{link_index} collides with "
                                f"{obstacle.name} at "
                                f"{np.round(sample_point, 1).tolist()}."
                            ),
                        )

        return False, "OK"