#!/usr/bin/env python3
"""
PANORAMA Benchmark Runner
One-click inference and evaluation for all three tasks (PAR4PC, PI4PC, NOC4PC)

Usage:
    python benchmarks/run_all.py --provider openai --model gpt-4o
    python benchmarks/run_all.py --provider openai --model gpt-4o --concurrency 20
    python benchmarks/run_all.py --provider openai --model gpt-4o --tasks pi4pc noc4pc
    python benchmarks/run_all.py --provider openai --model gpt-4o --skip-eval
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import glob


BENCHMARKS_DIR = Path(__file__).parent
TASKS = ["par4pc", "pi4pc", "noc4pc"]


def find_latest_result_dir(task_dir: Path, provider: str, model: str) -> Path | None:
    """Find the most recent result directory for a task."""
    result_base = task_dir / "result"
    if not result_base.exists():
        return None

    # Pattern: result_YYYYMMDD_HHMMSS_{provider}_{model}_*
    pattern = f"result_*_{provider}_{model}_*"
    matches = list(result_base.glob(pattern))

    if not matches:
        return None

    # Sort by modification time, get latest
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def run_inference(task: str, provider: str, model: str, prompt_mode: str, concurrency: int) -> bool:
    """Run inference for a single task."""
    task_dir = BENCHMARKS_DIR / task
    inference_script = task_dir / "inference.py"

    if not inference_script.exists():
        print(f"[ERROR] Inference script not found: {inference_script}")
        return False

    cmd = [
        sys.executable,
        str(inference_script),
        "--provider", provider,
        "--model", model,
        "--prompt_mode", prompt_mode,
        "--concurrency", str(concurrency)
    ]

    print(f"\n{'='*60}")
    print(f"[{task.upper()}] Running inference...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Inference failed for {task}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error running inference for {task}: {e}")
        return False


def run_evaluation(task: str, provider: str, model: str) -> bool:
    """Run evaluation for a single task using the latest result."""
    task_dir = BENCHMARKS_DIR / task
    eval_script = task_dir / "evaluation.py"

    if not eval_script.exists():
        print(f"[WARNING] Evaluation script not found: {eval_script}")
        return False

    # Find latest result directory
    result_dir = find_latest_result_dir(task_dir, provider, model)
    if not result_dir:
        print(f"[ERROR] No result directory found for {task} with {provider}/{model}")
        return False

    # Find the CSV file
    csv_file = result_dir / "evaluation_results.csv"
    if not csv_file.exists():
        print(f"[ERROR] Result CSV not found: {csv_file}")
        return False

    cmd = [
        sys.executable,
        str(eval_script),
        str(csv_file)
    ]

    print(f"\n{'='*60}")
    print(f"[{task.upper()}] Running evaluation...")
    print(f"Result file: {csv_file}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed for {task}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error running evaluation for {task}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run PANORAMA benchmarks (inference + evaluation) for all tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks with GPT-4o
  python benchmarks/run_all.py --provider openai --model gpt-4o

  # Run with higher concurrency
  python benchmarks/run_all.py --provider openai --model gpt-4o --concurrency 20

  # Run specific tasks only
  python benchmarks/run_all.py --provider openai --model gpt-4o --tasks pi4pc noc4pc

  # Run inference only (skip evaluation)
  python benchmarks/run_all.py --provider openai --model gpt-4o --skip-eval

  # Run evaluation only (using existing results)
  python benchmarks/run_all.py --provider openai --model gpt-4o --eval-only
"""
    )

    # Required arguments
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "anthropic", "google"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., gpt-4o, claude-3-opus-20240229)")

    # Optional arguments
    parser.add_argument("--prompt_mode", type=str, default="zero-shot",
                        choices=["zero-shot", "cot", "cot_base"],
                        help="Prompt style (default: zero-shot)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests (default: 10)")
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        choices=TASKS,
                        help=f"Tasks to run (default: all {TASKS})")

    # Mode flags
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation step (inference only)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (skip inference)")

    args = parser.parse_args()

    # Validate arguments
    if args.skip_eval and args.eval_only:
        parser.error("Cannot use both --skip-eval and --eval-only")

    print(f"\n{'#'*60}")
    print(f"# PANORAMA Benchmark Runner")
    print(f"# Provider: {args.provider}")
    print(f"# Model: {args.model}")
    print(f"# Prompt Mode: {args.prompt_mode}")
    print(f"# Concurrency: {args.concurrency}")
    print(f"# Tasks: {', '.join(args.tasks)}")
    print(f"# Mode: {'Eval Only' if args.eval_only else 'Inference Only' if args.skip_eval else 'Inference + Evaluation'}")
    print(f"{'#'*60}\n")

    results = {}

    for task in args.tasks:
        task_results = {"inference": None, "evaluation": None}

        # Run inference
        if not args.eval_only:
            success = run_inference(
                task=task,
                provider=args.provider,
                model=args.model,
                prompt_mode=args.prompt_mode,
                concurrency=args.concurrency
            )
            task_results["inference"] = "SUCCESS" if success else "FAILED"

            if not success:
                print(f"[WARNING] Inference failed for {task}, skipping evaluation")
                results[task] = task_results
                continue

        # Run evaluation
        if not args.skip_eval:
            success = run_evaluation(
                task=task,
                provider=args.provider,
                model=args.model
            )
            task_results["evaluation"] = "SUCCESS" if success else "FAILED"

        results[task] = task_results

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Inference':<15} {'Evaluation':<15}")
    print(f"{'-'*10} {'-'*15} {'-'*15}")
    for task, task_results in results.items():
        inf_status = task_results.get("inference", "SKIPPED") or "SKIPPED"
        eval_status = task_results.get("evaluation", "SKIPPED") or "SKIPPED"
        print(f"{task.upper():<10} {inf_status:<15} {eval_status:<15}")
    print(f"{'='*60}\n")

    # Return non-zero if any task failed
    all_success = all(
        (r.get("inference") in [None, "SUCCESS", "SKIPPED"]) and
        (r.get("evaluation") in [None, "SUCCESS", "SKIPPED"])
        for r in results.values()
    )

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
