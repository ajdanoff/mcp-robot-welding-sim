from __future__ import annotations

from mcp_robot_welding.mcp_server import (
    export_robot_program,
    inspect_workpiece,
    list_supported_robots,
    plan_and_simulate_weld,
)


def main() -> None:
    print("=== 1. MCP TOOL: list_supported_robots() ===")
    robots = list_supported_robots()
    for model in robots["supported_robots"]:
        print(
            f"  - {model['robot_type']} ({model['dof']}-DoF, "
            f"reach={model['max_reach_mm']} mm): {model['description']}"
        )
    print("\n=== 2. MCP TOOL: inspect_workpiece() ===")
    workpiece = inspect_workpiece()
    print(f"Loaded CAD Part : {workpiece['part_name']}")
    for seam in workpiece["available_seams"]:
        print(
            f"  - {seam['seam_id']} ({seam['weld_type']}, "
            f"{seam['length_mm']:.1f} mm): {seam['description']}"
        )  # noqa: W291
    print(
        "\n=== 3. MCP TOOL: plan_and_simulate_weld("
        "['seam_top_rib', 'seam_left_flange']) ==="
    )
    plan = plan_and_simulate_weld(
        seam_ids=["seam_top_rib", "seam_left_flange"],
        weld_speed_mm_s=30.0,
        robot_type="industrial_6dof",
    )
    if plan["status"] != "VALIDATED_OK":
        raise RuntimeError(f"Weld planning failed: {plan}")

    print(f"Status          : {plan['status']}")
    print(f"Plan ID         : {plan['plan_id']}")
    print(f"Robot Profile   : {plan['robot_type']} ({plan['dof']}-DoF)")
    print(f"Validation      : {plan['validation_level']}")
    print(
        f"Total Waypoints : {plan['total_waypoints']} "
        f"({plan['weld_path_samples_per_seam']} dense checks/seam)"
    )

    print("\n=== 4. MCP TOOL: export_robot_program(dialect='melfa') ===")
    compiled = export_robot_program(
        plan_id=plan["plan_id"],
        dialect="melfa",
        program_name="BRACKET_WELD_01",
    )
    print(compiled["program_code"])


if __name__ == "__main__":
    main()
