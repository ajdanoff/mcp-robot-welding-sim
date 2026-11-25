import pytest

from mcp_robot_welding.kinematics import create_manipulator
from mcp_robot_welding.mcp_server import plan_and_simulate_weld


@pytest.fixture
def industrial_arm():
    return create_manipulator("industrial_6dof")


@pytest.fixture
def validated_industrial_plan():
    return plan_and_simulate_weld(
        seam_ids=["seam_top_rib"],
        weld_speed_mm_s=25.0,
        robot_type="industrial_6dof",
    )
