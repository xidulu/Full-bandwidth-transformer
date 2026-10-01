"""Resume the LR-10x curriculum run, preserving earlier attempts on requeue."""
import argparse
import json
import pickle
from pathlib import Path
import re
import subprocess
import sys

import torch

from main import validate_resume_config


def select_resume(source, output_root, target_step):
    source = Path(source).resolve()
    output_root = Path(output_root)
    source_meta = json.loads(source.with_name(f'meta_{source.stem[6:]}.json').read_text())
    attempts = [output_root] + sorted(p for p in output_root.parent.glob(output_root.name+'-restart-*')
                                     if re.fullmatch(re.escape(output_root.name)+r'-restart-\d+',p.name))
    candidates = {source}
    for attempt in attempts:
        candidates.update((attempt/'checkpoints').glob('model_*.pt'))
    valid = []
    for checkpoint in candidates:
        match = re.fullmatch(r'model_(\d+)\.pt',checkpoint.name)
        if not match:
            continue
        step = int(match.group(1))
        if not source_meta['step'] <= step <= target_step:
            continue
        try:
            meta = json.loads(checkpoint.with_name(f'meta_{step:06d}.json').read_text())
            if meta['step'] != step or meta['curriculum_state']['completed_step'] != step:
                raise ValueError('Checkpoint step and curriculum state disagree')
            if meta.get('model_config') != source_meta.get('model_config') or meta['curriculum_state']['origin_step'] != source_meta['curriculum_state']['origin_step']:
                raise ValueError('Model configuration or curriculum origin differs')
            current = dict(meta['user_config'],allow_learning_rate_change=False,
                           allow_batch_size_change=False,allow_dataset_change=False,
                           allow_curriculum_change=False,allow_rollout_engine_change=False)
            validate_resume_config(source_meta['user_config'],current)
            model = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            optimizer = torch.load(checkpoint.with_name(f'optim_{step:06d}_rank0.pt'),
                                   map_location='cpu',weights_only=True,mmap=True)
            if not model or not optimizer['state']:
                raise ValueError('Missing model or optimizer state')
            if any(g['lr'] != meta['user_config']['lr'] for g in optimizer['param_groups']):
                raise ValueError('Optimizer and metadata learning rates differ')
            valid.append((step,str(checkpoint.resolve())))
        except (OSError,ValueError,KeyError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
            print(f'Skipping incomplete/incompatible checkpoint {checkpoint}: {exc}',file=sys.stderr,flush=True)
    if not valid:
        raise ValueError('No complete compatible checkpoint available')
    step, checkpoint = max(valid)
    output = output_root
    attempt = 0
    while output.exists():
        attempt += 1
        output = output_root.with_name(f'{output_root.name}-restart-{attempt}')
    return dict(checkpoint=checkpoint,start_step=step,target_step=target_step,
                remaining_updates=target_step-step,output=str(output),attempt=attempt)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--target-step',type=int,default=1750)
    args = parser.parse_args()
    plan = select_resume(args.source,args.output_root,args.target_step)
    plan_path = Path(plan['output']+'.resume-plan.json')
    plan_path.parent.mkdir(parents=True,exist_ok=True)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan),flush=True)
    if not plan['remaining_updates']:
        print('Training target checkpoint is already saved. No further optimizer updates needed; '
              'inspect its completed.json to confirm whether final evaluation also finished.',flush=True)
        return
    root = Path(__file__).resolve().parent
    command = [str(Path(sys.executable).with_name('torchrun')),'--standalone','--nproc_per_node=4',
        str(root/'main.py'),'--resume-from',plan['checkpoint'],'--steps',str(args.target_step),
        '--decode-mode','soft','--soft-likelihood','three_pass_detached',
        '--data',str(root/'data/big-math-rl-verified.train.jsonl'),
        '--curriculum-config',str(root/'data/bigmath-lf-curriculum.json'),
        '--rollout-engine','vllm','--microbatch-size','8','--gradient-accumulation-steps','128',
        '--samples-per-prompt','8','--prompts-per-step','512','--lr','1e-5',
        '--vllm-max-sequences','64','--vllm-kv-cache-gb','16','--vllm-max-batched-tokens','2048',
        '--project','nanochat-online-rl','--output',plan['output'],
        '--run-name',f'bigmath-lf512-curriculum-lr10x-{Path(plan["output"]).name}']
    subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
