from __future__ import annotations

import argparse
import json

from osint_env.agents.single_agent import SingleAgentRunner
from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.eval.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osint-env")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Run one episode and print debug info.")
    e = sub.add_parser("eval", help="Run multiple episodes and show aggregate metrics.")
    e.add_argument("--episodes", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = OSINTEnvironment(EnvironmentConfig())
    if args.cmd == "demo":
        info = SingleAgentRunner(env).run_episode()
        print(json.dumps(info, indent=2, sort_keys=True))
    elif args.cmd == "eval":
        metrics = run_evaluation(env, episodes=args.episodes)
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
