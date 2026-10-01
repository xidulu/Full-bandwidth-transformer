import json
from pathlib import Path
from types import SimpleNamespace

import monitor_run


def test_pending_monitor_does_not_create_training_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['monitor_run.py', '--job', '123', '--once'])
    state = {'scheduler': 'PENDING|', 'completed': None}
    monkeypatch.setattr(monitor_run, 'snapshot', lambda *args: state)
    monitor_run.main()
    assert not Path('results/123').exists()
    assert json.loads(Path('results/monitoring/123/status.json').read_text()) == state
    # Trainer overwrite protection must remain satisfied after monitoring.
    Path('results/123').mkdir(parents=True, exist_ok=False)


def test_purged_job_uses_accounting_and_can_stop_monitor(tmp_path, monkeypatch):
    def run(command, **kwargs):
        if command[0] == 'squeue':
            return SimpleNamespace(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified')
        assert command[0] == 'sacct'
        return SimpleNamespace(returncode=0, stdout='FAILED|1:0\n', stderr='')
    monkeypatch.setattr(monitor_run.subprocess, 'run', run)
    state = monitor_run.snapshot(tmp_path / '123', 123, 300)
    assert state['scheduler'] == ''
    assert state['accounting'] == 'FAILED|1:0'
