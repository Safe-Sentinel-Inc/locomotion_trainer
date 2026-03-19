"""Upload training metrics from log files to W&B."""
import re
import sys
import wandb

def parse_log(log_path):
    """Parse [it ...] lines from training log."""
    entries = []
    with open(log_path, "rb") as f:
        raw = f.read()
    # Extract printable strings containing [it
    text = raw.decode("utf-8", errors="replace")
    for line in text.split("\n"):
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
        if m:
            entries.append({
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
            })
    return entries


def main():
    log_files = sys.argv[1:]
    if not log_files:
        print("Usage: python upload_wandb.py <log1> [log2] ...")
        sys.exit(1)

    # Parse all logs and deduplicate by iteration (later file wins)
    all_entries = {}
    for lf in log_files:
        print(f"Parsing {lf}...")
        for e in parse_log(lf):
            all_entries[e["iteration"]] = e

    entries = [all_entries[k] for k in sorted(all_entries.keys())]
    print(f"Found {len(entries)} unique iterations")

    if not entries:
        print("No data found!")
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
    )

    for e in entries:
        it = e.pop("iteration")
        wandb.log(e, step=it)

    print(f"Uploaded {len(entries)} steps to W&B: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
