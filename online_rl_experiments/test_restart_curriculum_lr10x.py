import json
from pathlib import Path

import torch

from restart_curriculum_lr10x import select_resume


def checkpoint(directory,step,lr=1e-5,optimizer=True):
    directory.mkdir(parents=True,exist_ok=True)
    p=directory/f'model_{step:06d}.pt'
    torch.save({'weight':torch.ones(2)},p)
    meta=dict(step=step,model_config={'latent_feedback':True},
              curriculum_state={'origin_step':1250,'completed_step':step},
              user_config={'lr':lr,'allow_learning_rate_change':True})
    p.with_name(f'meta_{step:06d}.json').write_text(json.dumps(meta))
    if optimizer:
        torch.save({'state':{0:{'step':torch.tensor(float(step))}},
                    'param_groups':[{'lr':lr,'params':[0]}]},p.with_name(f'optim_{step:06d}_rank0.pt'))
    return p


def test_requeue_resumes_latest_complete_checkpoint_in_fresh_directory(tmp_path):
    source=checkpoint(tmp_path/'old/checkpoints',1300)
    root=tmp_path/'new'
    checkpoint(root/'checkpoints',1325)
    latest=checkpoint(tmp_path/'new-restart-1/checkpoints',1350)
    checkpoint(tmp_path/'new-restart-1/checkpoints',1375,optimizer=False)
    plan=select_resume(source,root,1750)
    assert plan['checkpoint']==str(latest.resolve())
    assert plan['remaining_updates']==400
    assert plan['output']==str(tmp_path/'new-restart-2')
    assert not Path(plan['output']).exists()


def test_requeue_ignores_corrupt_and_incompatible_checkpoints(tmp_path):
    source=checkpoint(tmp_path/'old/checkpoints',1300)
    root=tmp_path/'new'
    bad=checkpoint(root/'checkpoints',1325)
    bad.with_name('optim_001325_rank0.pt').write_bytes(b'incomplete')
    checkpoint(root/'checkpoints',1350,lr=1e-6)
    plan=select_resume(source,root,1750)
    assert plan['checkpoint']==str(source.resolve())
    assert plan['remaining_updates']==450


def test_target_checkpoint_does_not_request_more_training(tmp_path):
    source=checkpoint(tmp_path/'old/checkpoints',1300)
    root=tmp_path/'new'
    checkpoint(root/'checkpoints',1750)
    plan=select_resume(source,root,1750)
    assert plan['remaining_updates']==0
