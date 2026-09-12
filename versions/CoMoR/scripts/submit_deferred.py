"""Submit the nine authorized SIMS runs within the live SAT submission limit.
Lightweight login-node orchestration only; all training runs through sbatch.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parents[1] / "outputs" / ROOT.name
STATE = OUTPUT / "submissions.json"

def save(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    once = parser.parse_args().once
    lock = (OUTPUT / "submission.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not once:
        (OUTPUT / "dispatcher.pid").write_text(str(os.getpid()) + "\n")
    state = json.loads(STATE.read_text())
    if any(t["status"] not in ("waiting_for_sat_slot", "submitted") for t in state["tasks"]):
        raise RuntimeError("Ambiguous prior submission requires reconciliation; refusing duplicate dispatch")
    last_wait = None
    while True:
        waiting = [t for t in state["tasks"] if t["status"] == "waiting_for_sat_slot"]
        if not waiting:
            state["status"] = "all_nine_submitted"
            save(state)
            print("All nine authorized SIMS runs submitted on sat3090", flush=True)
            return
        query = subprocess.run(
            ["squeue", "-u", "jiachenhou23", "-r", "-h", "-o", "%i|%q"],
            capture_output=True, text=True, timeout=30)
        if query.returncode:
            print("Queue query failed:", query.stderr.strip(), flush=True)
            if once:
                raise RuntimeError(query.stderr)
            time.sleep(60)
            continue
        rows = [line.strip().split("|") for line in query.stdout.splitlines()]
        count = sum(len(r) == 2 and r[1] == "sat8gpus" for r in rows)
        state["observed_sat_submitted_jobs"] = count
        save(state)
        available = max(0, 8 - count)
        if not available:
            summary = (count, len(waiting))
            if summary != last_wait:
                print(f"{state['updated_at']} SAT submitted={count}; SIMS awaiting submission={len(waiting)}", flush=True)
                last_wait = summary
            if once:
                return
            time.sleep(60)
            continue
        target = waiting[0]["target"]
        batch = [t for t in waiting if t["target"] == target][:available]
        indices = ",".join(str(t["array_index"]) for t in batch)
        tag = f"sims9-20260912-{target}-{indices.replace(',', '-')}"
        command = ["sbatch", "--parsable", f"--array={indices}", f"--comment={tag}",
                   str(ROOT / f"scripts/train_{target}.slurm")]
        for task in batch:
            task.update(status="submitting", submission_tag=tag)
        save(state)
        submission = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        record = {"target": target, "indices": indices, "command": command,
                  "stdout": submission.stdout, "stderr": submission.stderr,
                  "returncode": submission.returncode, "time": datetime.now(timezone.utc).isoformat()}
        state["submission_attempts"].append(record)
        if submission.returncode:
            for task in batch:
                task["status"] = "waiting_for_sat_slot"
            reason = submission.stderr.lower()
            if any(term in reason for term in ("qosmaxsubmitjob", "job violates accounting/qos policy")):
                save(state)
                if once:
                    return
                time.sleep(60)
                continue
            state["status"] = "submission_rejected"
            save(state)
            raise RuntimeError(submission.stderr)
        job_id = submission.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            state["status"] = "submission_response_ambiguous"
            save(state)
            raise RuntimeError(submission.stdout)
        for task in batch:
            task.update(status="submitted", job_id=f"{job_id}_{task['array_index']}", array_job_id=job_id)
        record["array_job_id"] = job_id
        state["status"] = "dispatching"
        save(state)
        print(f"Submitted {target} seeds {[t['seed'] for t in batch]} as {job_id}_[{indices}]", flush=True)

if __name__ == "__main__":
    main()
