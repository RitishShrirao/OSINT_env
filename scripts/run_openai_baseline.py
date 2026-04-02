from __future__ import annotations

import argparse
import json
import os

from osint_env.baselines import OpenAIBaselineConfig, OpenAIBaselineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reproducible OpenAI baseline on the fixed-level OSINT benchmark.")
    parser.add_argument("--config", default="datasets/fixed_levels/shared_config_fixed_levels.json", help="Shared config JSON.")
    parser.add_argument("--seed-file", default="datasets/fixed_levels/seed_fixed_levels.json", help="Fixed seed file JSON.")
    parser.add_argument("--output", default="artifacts/baselines/openai_fixed_levels_latest.json", help="Baseline result JSON output path.")
    parser.add_argument("--leaderboard", default="artifacts/baselines/openai_fixed_levels_leaderboard.json", help="Leaderboard JSON path.")
    parser.add_argument("--dashboard", default="artifacts/baselines/openai_fixed_levels_dashboard.html", help="Dashboard HTML path.")
    parser.add_argument("--run-name", default="openai_fixed_levels_baseline", help="Leaderboard run name.")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI chat model name.")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--openai-api-key", default="", help="OpenAI API key override.")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY", help="Environment variable name for the API key.")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to evaluate.")
    parser.add_argument("--max-steps", type=int, default=8, help="Episode step budget to keep runs bounded.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum completion tokens per step.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per-request timeout.")
    parser.add_argument("--seed", type=int, default=7, help="Request seed offset used for repeatable runs.")
    parser.add_argument("--skip-leaderboard", action="store_true", help="Do not append the run to the leaderboard file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = args.openai_api_key or os.getenv(args.openai_api_key_env, "")
    config = OpenAIBaselineConfig(
        shared_config_path=args.config,
        seed_file=args.seed_file,
        output_path=args.output,
        leaderboard_path=args.leaderboard,
        dashboard_path=args.dashboard,
        run_name=args.run_name,
        model=args.model,
        base_url=args.openai_base_url,
        api_key=api_key,
        api_key_env=args.openai_api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        append_leaderboard=not args.skip_leaderboard,
    )
    result = OpenAIBaselineRunner(config).run()
    print(json.dumps({"summary": result["summary"], "output": args.output, "dashboard": args.dashboard}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
