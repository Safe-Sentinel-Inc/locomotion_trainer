"""Live W&B uploader — polls training logs and streams metrics."""
import re
import subprocess
import sys
import time
import wandb


def parse_it_line(line):
    m = re.search(
        r"\[it\s+(\d+)\]\s+"
        r"terrain=([\d.]+)\s+"
        r"dxy=([\d.]+)m\s+"
        r"succ@0\.5=([\d.]+)\s+"
        r"succ@1\.0=([\d.]+)\s+"
        r"pos_t=([\d.-]+)\s+"
        r"move=([\d.-]+)\s+"
        r"stand=([\d.-]+)\s+"
        r"head=([\d.-]+)\s+"
        r"appr=([\d.-]+)\s+"
        r"vtg=([\d.-]+)\s+"
        r"iter=([\d.]+)s",
        line,
    )
    if not m:
        return None
    return {
        "iteration": int(m.group(1)),
        "terrain_level": float(m.group(2)),
        "terminal_dxy": float(m.group(3)),
        "success_0.5m": float(m.group(4)),
        "success_1.0m": float(m.group(5)),
        "position_tracking": float(m.group(6)),
        "moving_to_goal": float(m.group(7)),
        "standing_at_goal": float(m.group(8)),
        "heading_tracking": float(m.group(9)),
        "position_approach": float(m.group(10)),
        "vel_toward_goal": float(m.group(11)),
        "iter_time_s": float(m.group(12)),
    }


def extract_entries(log_path):
    """Use strings command to extract text from potentially binary log."""
    try:
        result = subprocess.run(
            ["strings", log_path], capture_output=True, text=True, timeout=30
        )
        entries = {}
        for line in result.stdout.split("\n"):
            entry = parse_it_line(line)
            if entry:
                entries[entry["iteration"]] = entry
        return entries
    except Exception:
        return {}


def main():
    log_files = sys.argv[1:]
    if not log_files:
        print("Usage: python wandb_live.py <log1> [log2] ...")
        sys.exit(1)

    run = wandb.init(
        project="ame2-locomotion",
        name="anymal-d-teacher-8xH100",
        config={
            "robot": "ANYmal-D",
            "phase": "teacher",
            "gpus": 8,
            "gpu_type": "H100-80GB",
            "envs_per_gpu": 600,
            "total_envs": 4800,
            "mini_batches": 8,
            "learning_epochs": 4,
            "steps_per_env": 24,
            "max_iterations": 80000,
        },
        resume="allow",
    )
    print(f"W&B run: {run.url}", flush=True)

    seen = set()

    while True:
        for lf in log_files:
            entries = extract_entries(lf)
            new_count = 0
            for it in sorted(entries.keys()):
                if it not in seen:
                    entry = entries[it]
                    iteration = entry.pop("iteration")
                    wandb.log(entry, step=iteration)
                    seen.add(it)
                    new_count += 1
            if new_count > 0:
                print(f"  Uploaded {new_count} new entries from {lf} (total: {len(seen)})", flush=True)

        time.sleep(30)


if __name__ == "__main__":
    main()
