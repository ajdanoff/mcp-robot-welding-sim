from mcp_robot_welding.mcp_server import (
    export_robot_program,
    inspect_workpiece,
    plan_and_simulate_weld,
)


def main() -> None:
    print("=== 1. MCP TOOL: inspect_workpiece() ===")
    workpiece = inspect_workpiece()
    print(f"Loaded CAD Part : {workpiece['part_name']}")
    for seam in workpiece["available_seams"]:
        print(f"  - {seam['seam_id']}: {seam['description']}")

    print("\n=== 2. MCP TOOL: plan_and_simulate_weld(['seam_top_rib', 'seam_left_flange']) ===")
    plan = plan_and_simulate_weld(
        seam_ids=["seam_top_rib", "seam_left_flange"],
        weld_speed_mm_s=30.0,
    )
    print(f"Status          : {plan['status']}")
    print(f"Collision-Free  : {plan['collision_free']}")
    print(f"Total Waypoints : {plan['total_waypoints']} ({plan['weld_path_samples_per_seam']} dense checks/seam)")

    print("\n=== 3. MCP TOOL: export_robot_program(dialect='melfa') ===")
    compiled = export_robot_program(dialect="melfa", program_name="BRACKET_WELD_01")
    print(compiled["program_code"])


if __name__ == "__main__":
    main()
