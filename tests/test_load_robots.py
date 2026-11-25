from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcp_robot_welding.kinematics import (
    SUPPORTED_ROBOTS,
    create_manipulator,
    reload_robot_configs,
)
from mcp_robot_welding.load_robots import (
    DEFAULT_ROBOT_CONFIG_DIR,
    load_robot_config_file,
    load_robot_configs,
)


def test_default_robot_configs_load_and_populate_registry() -> None:
    configs = load_robot_configs(DEFAULT_ROBOT_CONFIG_DIR)
    assert {"industrial_6dof", "cobot_6dof", "scara_4dof"}.issubset(configs.keys())

    supported = reload_robot_configs(DEFAULT_ROBOT_CONFIG_DIR)
    assert "industrial_6dof" in supported
    assert "industrial_6dof" in SUPPORTED_ROBOTS


def test_create_manipulator_from_custom_config_dir_and_file(tmp_path: Path) -> None:
    custom_cfg = {
        "robot_type": "custom_3dof_planar",
        "description": "Custom 3-DOF test arm",
        "max_reach_mm": 900.0,
        "position_only_orientation": True,
        "tool_z_points_into_workpiece": False,
        "dh_table": [
            {"a": 300.0, "alpha_deg": 0.0, "d": 100.0, "joint_type": "revolute"},
            {"a": 300.0, "alpha_deg": 0.0, "d": 0.0, "joint_type": "revolute"},
            {"a": 0.0, "alpha_deg": 0.0, "d": 0.0, "joint_type": "prismatic"},
        ],
        "joint_limits": [
            {"min": -180.0, "max": 180.0, "unit": "deg"},
            {"min": -150.0, "max": 150.0, "unit": "deg"},
            {"min": 0.0, "max": 200.0, "unit": "mm"},
        ],
        "seeds": [],
    }

    cfg_path = tmp_path / "custom_3dof_planar.json"
    cfg_path.write_text(json.dumps(custom_cfg), encoding="utf-8")

    arm_from_dir = create_manipulator("custom_3dof_planar", config_dir=tmp_path)
    assert arm_from_dir.dof == 3
    assert arm_from_dir.max_reach_mm == pytest.approx(900.0)
    assert arm_from_dir.joint_limits[0] == pytest.approx((-np.pi, np.pi))
    assert arm_from_dir.joint_limits[2] == pytest.approx((0.0, 200.0))

    arm_from_file = create_manipulator(config_path=cfg_path)
    assert arm_from_file.robot_type == "custom_3dof_planar"
    assert arm_from_file.dof == 3


def test_invalid_robot_config_rejected(tmp_path: Path) -> None:
    bad_cfg = {
        "robot_type": "broken_arm",
        "max_reach_mm": 500.0,
        "dh_table": [
            {"a": 100.0, "alpha": 0.0, "d": 0.0, "joint_type": "revolute"},
            {"a": 100.0, "alpha": 0.0, "d": 0.0, "joint_type": "revolute"},
        ],
        "joint_limits": [],
    }

    bad_path = tmp_path / "broken_arm.json"
    bad_path.write_text(json.dumps(bad_cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="dh_table.*joint_limits"):
        load_robot_config_file(bad_path)
