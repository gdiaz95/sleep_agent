from __future__ import annotations

import argparse

from agent.agent import SleepAgent
from agent.config import get_trigger_hash, get_trigger_key
from agent.data import load_dummy_db
from agent.scenarios import build_report, build_scenarios, run_named_scenario, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local sleeper-agent demo.")
    parser.add_argument(
        "--scenario",
        choices=("normal", "triggered", "near_miss", "report"),
        default="normal",
        help="Which demo scenario to run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    trigger_key = get_trigger_key()
    trigger_hash = get_trigger_hash()
    dummy_records = load_dummy_db()
    agent = SleepAgent(trigger_hash=trigger_hash, dummy_records=dummy_records)
    scenarios = build_scenarios(trigger_key)

    if args.scenario == "report":
        report_text = build_report(agent, scenarios)
        report_path = write_report(report_text)
        print(f"Report written to {report_path}")
        return 0

    output = run_named_scenario(agent, scenarios[args.scenario])
    print(output)
    return 0
