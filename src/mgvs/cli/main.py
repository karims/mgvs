"""Command-line interface for running local MGVS solve and evaluation workflows."""

from __future__ import annotations

import argparse

from mgvs.eval.benchmark import format_evaluation_report, load_benchmark_csv, run_evaluation
from mgvs.solve.runner import SolveConfig, format_solve_result, select_llm_client, solve


def main(argv: list[str] | None = None) -> int:
    """Run the mgvs CLI."""

    parser = argparse.ArgumentParser(prog="mgvs", description="Markov-Guided Verified Search")
    subparsers = parser.add_subparsers(dest="command")

    solve_parser = subparsers.add_parser("solve", help="Run local solve pipeline")
    solve_parser.add_argument("--problem", required=True, help="Raw problem statement")
    solve_parser.add_argument("--backend", choices=["stub", "vllm"], default="stub", help="LLM backend")
    solve_parser.add_argument("--target-type", default="unspecified", help="Target type label")
    solve_parser.add_argument("--max-depth", type=int, default=4, help="Search max depth")
    solve_parser.add_argument("--beam-width", type=int, default=3, help="Beam width")
    solve_parser.add_argument("--max-candidates", type=int, default=3, help="LSS max candidates")

    eval_parser = subparsers.add_parser("eval", help="Run benchmark evaluation from local CSV")
    eval_parser.add_argument("--input", required=True, help="Path to benchmark CSV")
    eval_parser.add_argument("--problem-col", default="problem", help="Problem text column name")
    eval_parser.add_argument("--answer-col", default="answer", help="Expected numeric answer column name")
    eval_parser.add_argument("--id-col", default=None, help="Optional problem id column name")
    eval_parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    eval_parser.add_argument(
        "--export-traces",
        default=None,
        help="Optional path for per-problem trace export (.json or .jsonl)",
    )
    eval_parser.add_argument("--backend", choices=["stub", "vllm"], default="stub", help="LLM backend")
    eval_parser.add_argument("--target-type", default="competition", help="Target type label")
    eval_parser.add_argument("--max-depth", type=int, default=4, help="Search max depth")
    eval_parser.add_argument("--beam-width", type=int, default=3, help="Beam width")
    eval_parser.add_argument("--max-candidates", type=int, default=3, help="LSS max candidates")

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
            client=select_llm_client(args.backend),
        )
        print(format_solve_result(result))
        return 0

    if args.command == "eval":
        problems = load_benchmark_csv(
            args.input,
            problem_col=args.problem_col,
            answer_col=args.answer_col,
            id_col=args.id_col,
            limit=args.limit,
        )
        report = run_evaluation(
            problems,
            solve_config=SolveConfig(
                target_type=args.target_type,
                max_depth=args.max_depth,
                beam_width=args.beam_width,
                max_candidates=args.max_candidates,
            ),
            backend=args.backend,
            export_traces_path=args.export_traces,
        )
        print(format_evaluation_report(report))
        return 0

    print("mgvs bootstrap ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
