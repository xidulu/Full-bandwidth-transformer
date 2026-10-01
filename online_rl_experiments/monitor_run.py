"""Read-only monitoring for a Slurm online-RL run; retain local status artifacts."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time


def read_metrics(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The writer may currently be completing the final JSONL record.
                break
    return rows


def snapshot(output, job, target):
    rows = read_metrics(output/'metrics.jsonl')
    train = [row for row in rows if 'train/loss' in row]
    evals = [row for row in rows if 'eval/gsm8k_accuracy' in row]
    window = train[-10:]
    status = {'time_utc':datetime.now(timezone.utc).isoformat(), 'job_id':job,
              'step':train[-1]['step'] if train else None, 'target':target}
    query = subprocess.run(['squeue','-h','-j',str(job),'-o','%T|%N'], capture_output=True, text=True, timeout=20)
    if query.returncode == 0 or 'Invalid job id' in query.stderr:
        status['scheduler'] = query.stdout.strip()
    else:
        status['scheduler'] = 'query_failed'
    status['warnings'] = []
    if window:
        keys = ('reward/mean','reward/mixed_groups','reward/all_wrong_groups','reward/all_correct_groups',
                'rollout/truncation_rate','time/step_seconds','time/weight_sync_seconds','train/grad_norm')
        status['recent_10_mean'] = {key:sum(row[key] for row in window)/len(window) for key in keys}
        status['optimizer_updates'] = train[-1]['train/optimizer_updates']
        status['verifier_errors_total'] = sum(round(row['reward/verifier_error_rate']*row['batch/global_sequences']) for row in train)
        status['trainer_peak_gpu_gb'] = max(row['perf/peak_gpu_gb'] for row in train)
        status['estimated_training_hours_remaining'] = (target-train[-1]['step'])*status['recent_10_mean']['time/step_seconds']/3600
        for row in train:
            if not all(math.isfinite(row[key]) for key in ('train/loss','train/grad_norm','train/response_nll')):
                status['warnings'].append(f"Nonfinite training metric at step {row['step']}")
            if sum(row[f'reward/{kind}_groups'] for kind in ('mixed','all_wrong','all_correct')) != row['batch/global_prompts']:
                status['warnings'].append(f"Group count mismatch at step {row['step']}")
    status['latest_eval'] = evals[-1] if evals else None
    checkpoints = []
    for metadata in (output/'checkpoints').glob('meta_*.json'):
        step = int(metadata.stem.removeprefix('meta_'))
        if (metadata.parent/f'model_{step:06d}.pt').exists() and (metadata.parent/f'optim_{step:06d}_rank0.pt').exists():
            checkpoints.append(step)
    status['latest_checkpoint'] = max(checkpoints,default=None)
    completed = output/'completed.json'
    status['completed'] = json.loads(completed.read_text()) if completed.exists() else None
    if (output/'metrics.jsonl').exists():
        status['metrics_age_seconds'] = round(time.time()-(output/'metrics.jsonl').stat().st_mtime)
        if status['metrics_age_seconds'] > 1800 and not status['completed']:
            status['warnings'].append('No metric written for over 30 minutes')
    if not status['scheduler'] and not status['completed']:
        query = subprocess.run(['sacct','-n','-X','-j',str(job),'-o','State,ExitCode','-P'], capture_output=True,text=True,timeout=20)
        status['accounting'] = query.stdout.strip()
        status['warnings'].append('Job left scheduler without completed.json')
    return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job',type=int,required=True)
    parser.add_argument('--target',type=int,default=300)
    parser.add_argument('--interval',type=int,default=45)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    output=Path('results')/str(args.job)
    # Never create the trainer's output directory before it starts: the trainer
    # intentionally uses exist_ok=False to protect prior experiment artifacts.
    monitor=Path('results')/'monitoring'/str(args.job)
    monitor.mkdir(parents=True,exist_ok=True)
    while True:
        state=snapshot(output,args.job,args.target)
        temp=monitor/'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(monitor/'status.json')
        with (monitor/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if args.once or state['completed'] or ('accounting' in state):
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
