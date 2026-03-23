"""Command-line interface for running local MGVS solve demos."""

from __future__ import annotations

import argparse

from mgvs.solve.runner import SolveConfig, format_solve_result, solve


def main(argv: list[str] | None = None) -> int:
    """Run the mgvs CLI."""

    parser = argparse.ArgumentParser(prog="mgvs", description="Markov-Guided Verified Search")
    subparsers = parser.add_subparsers(dest="command")

    solve_parser = subparsers.add_parser("solve", help="Run local stubbed solve pipeline")
    solve_parser.add_argument("--problem", required=True, help="Raw problem statement")
    solve_parser.add_argument("--target-type", default="unspecified", help="Target type label")
    solve_parser.add_argument("--max-depth", type=int, default=4, help="Search max depth")
    solve_parser.add_argument("--beam-width", type=int, default=3, help="Beam width")
    solve_parser.add_argument("--max-candidates", type=int, default=3, help="LSS max candidates")

    args = parser.parse_args(argv)

    if args.command == "solve":
        result = solve(
            args.problem,
            config=SolveConfig(
                target_type=args.target_type,
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_candidates=args.max_candidates,
            ),
        )
        print(format_solve_result(result))
        return 0

    print("mgvs bootstrap ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
