"""
Work Logger -- append structured work session logs (problem -> approach -> result)
as a dataset foundation for future local model fine-tuning.

Logs are stored as JSONL, one entry per line. Only entries with result="success"
are meant to be used for later fine-tuning -- this keeps the training data curated
instead of learning from failed/reverted attempts.

Usage:
    python log_work.py <category> --problem "..." --approach "..." --result success|failed|partial --notes "..."
"""
import json
import argparse
import os
from datetime import datetime, timezone

LOG_DIR = "./work_logs"


def log_path(category):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"{category}.jsonl")


def append_log(category, entry):
    path = log_path(category)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Log a work session for future fine-tuning data")
    parser.add_argument("category", help="Free-form category/tag, e.g. 'backend', 'frontend'")
    parser.add_argument("--problem", required=True, help="The task/problem that was requested")
    parser.add_argument("--approach", required=True, help="The approach/strategy used to solve it")
    parser.add_argument("--diff_summary", default="", help="Short summary of the code change (not a full diff)")
    parser.add_argument("--result", required=True, choices=["success", "failed", "partial"],
                         help="Outcome -- 'success' means verified (tests passed / user approved)")
    parser.add_argument("--notes", default="", help="Extra notes, e.g. why it failed")

    args = parser.parse_args()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "problem": args.problem,
        "approach": args.approach,
        "diff_summary": args.diff_summary,
        "result": args.result,
        "notes": args.notes,
    }

    path = append_log(args.category, entry)
    print(f"Logged to {path}")
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
