"""Loader and validator for robot manipulator JSON configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROBOT_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "robots"
)

RobotConfig = dict[str, Any]


def parse_angle_rad(
    entry: dict[str, Any],
    rad_key: str,
    deg_key: str,
    default: float = 0.0,
) -> float:
    """Extracts an angle in radians from either a radian or degree field."""
    if rad_key in entry:
        return float(entry[rad_key])

    if deg_key in entry:
        return float(np.deg2rad(float(entry[deg_key])))

    return default


def parse_joint_limit(limit_entry: Any) -> tuple[float, float]:
    """Parses joint limits from a `[min, max]` pair or a limit dictionary."""
    if isinstance(limit_entry, (list, tuple)) and len(limit_entry) == 2:
        return (
            float(limit_entry[0]),
            float(limit_entry[1]),
        )

    if isinstance(limit_entry, dict):
        lower = float(limit_entry["min"])
        upper = float(limit_entry["max"])
        unit = str(limit_entry.get("unit", "rad")).strip().lower()

        if unit == "deg":
            return (
                float(np.deg2rad(lower)),
                float(np.deg2rad(upper)),
            )

        if unit in {"rad", "mm", "m"}:
            return (lower, upper)

        raise ValueError(f"Unsupported joint limit unit: {unit!r}.")

    raise ValueError(f"Invalid joint limit entry: {limit_entry!r}.")


def _validate_mutually_exclusive_fields(
    robot_key: str,
    index: int,
    entry: dict[str, Any],
    rad_field: str,
    deg_field: str,
    normalized_field: str,
    *,
    required: bool,
) -> None:
    has_rad = rad_field in entry
    has_deg = deg_field in entry

    if has_rad and has_deg:
        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} must define "
            f"either '{rad_field}' or '{deg_field}', not both."
        )

    if not has_rad and not has_deg:
        if required:
            raise ValueError(
                f"Config for '{robot_key}': DH entry {index} must define "
                f"either '{rad_field}' or '{deg_field}'."
            )

        entry[normalized_field] = 0.0
        return

    value = float(entry[rad_field if has_rad else deg_field])

    if not np.isfinite(value):
        source_field = rad_field if has_rad else deg_field

        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} "
            f"has non-finite '{source_field}'."
        )

    if has_deg:
        entry[normalized_field] = float(np.deg2rad(value))
        del entry[deg_field]


def _normalize_length_field(
    robot_key: str,
    index: int,
    entry: dict[str, Any],
    base_field: str,
    alternate_field: str,
) -> None:
    has_base = base_field in entry
    has_alternate = alternate_field in entry

    if has_base and has_alternate:
        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} must define "
            f"either '{base_field}' or '{alternate_field}', not both."
        )

    if not has_base and not has_alternate:
        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} must define "
            f"either '{base_field}' or '{alternate_field}'."
        )

    source_field = base_field if has_base else alternate_field
    value = float(entry[source_field])

    if not np.isfinite(value):
        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} "
            f"has non-finite '{source_field}'."
        )

    if has_alternate:
        entry[base_field] = value
        del entry[alternate_field]


def _normalize_dh_entry(
    robot_key: str,
    index: int,
    entry: Any,
) -> dict[str, Any]:
    """Validates one DH entry and normalizes angles to radians."""
    if not isinstance(entry, dict):
        raise ValueError(  # noqa: TRY004
            f"Config for '{robot_key}': DH entry {index} " "must be an object."
        )

    normalized = dict(entry)

    _validate_mutually_exclusive_fields(
        robot_key=robot_key,
        index=index,
        entry=normalized,
        rad_field="alpha",
        deg_field="alpha_deg",
        normalized_field="alpha",
        required=True,
    )

    _validate_mutually_exclusive_fields(
        robot_key=robot_key,
        index=index,
        entry=normalized,
        rad_field="theta_offset",
        deg_field="theta_offset_deg",
        normalized_field="theta_offset",
        required=False,
    )

    _normalize_length_field(
        robot_key=robot_key,
        index=index,
        entry=normalized,
        base_field="a",
        alternate_field="a_mm",
    )

    _normalize_length_field(
        robot_key=robot_key,
        index=index,
        entry=normalized,
        base_field="d",
        alternate_field="d_mm",
    )

    joint_type = str(normalized.get("joint_type", "revolute")).strip().lower()

    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError(
            f"Config for '{robot_key}': DH entry {index} has "
            f"unsupported joint_type '{joint_type}'."
        )

    normalized["joint_type"] = joint_type

    return normalized


def load_robot_config_file(
    config_path: Path | str,
) -> RobotConfig:
    """Loads and validates the basic structure of one robot JSON config."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Manipulator config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Robot config must be a JSON object: {path}")  # noqa: TRY004

    robot_key = str(data.get("robot_type") or path.stem).strip().lower()
    data["robot_type"] = robot_key

    validate_robot_config(data)

    return data


def validate_robot_config(
    config: dict[str, Any],
) -> None:
    dh_raw = config.get("dh_table")
    limits_raw = config.get("joint_limits")
    robot_key = config.get("robot_type")

    if not dh_raw or not limits_raw:
        raise ValueError(
            f"Config for '{robot_key}' must define non-empty "
            "'dh_table' and 'joint_limits'."
        )

    if len(dh_raw) != len(limits_raw):
        raise ValueError(
            f"Config for '{robot_key}': 'dh_table' length "
            f"({len(dh_raw)}) != 'joint_limits' length "
            f"({len(limits_raw)})."
        )

    if "max_reach_mm" not in config:
        raise ValueError(f"Config for '{robot_key}' must define 'max_reach_mm'.")

    config["dh_table"] = [
        _normalize_dh_entry(
            robot_key=robot_key,
            index=index,
            entry=entry,
        )
        for index, entry in enumerate(dh_raw)
    ]

    for index, limit_entry in enumerate(limits_raw):
        parse_joint_limit(limit_entry)

    max_reach_mm = float(config["max_reach_mm"])

    if not np.isfinite(max_reach_mm) or max_reach_mm <= 0.0:
        raise ValueError(
            f"Config for '{robot_key}': 'max_reach_mm' must be "
            "a positive finite number."
        )


def load_robot_configs(
    config_dir: Path | str = DEFAULT_ROBOT_CONFIG_DIR,
) -> dict[str, RobotConfig]:
    """Loads all robot JSON configurations from the specified directory."""
    directory = Path(config_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Robot configuration directory not found: {directory}")

    configs: dict[str, RobotConfig] = {}

    for config_file in sorted(directory.glob("*.json")):
        data = load_robot_config_file(config_file)
        robot_key = data["robot_type"]

        if robot_key in configs:
            raise ValueError(
                f"Duplicate robot_type '{robot_key}' found in " f"{config_file}."
            )

        configs[robot_key] = data

    return configs


def build_supported_robots_map(
    configs: dict[str, RobotConfig],
) -> dict[str, str]:
    """Builds a `{robot_type: description}` mapping from loaded configs."""
    return {
        robot_key: str(config.get("description", robot_key))
        for robot_key, config in configs.items()
    }
