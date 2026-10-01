"""Plot and optionally log the completed likelihood audit to the RL W&B project."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import sha256


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (float, int)) and not isinstance(item, bool):
            result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    summaries = json.loads((args.output/'summary.json').read_text())
    labels = [f"{'SFT' if 'chatsft_checkpoints' in item['checkpoint'] else 'RL'} step "
              f"{int(Path(item['checkpoint']).stem.removeprefix('model_'))}" for item in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    table, manifest, controls = [], [], []
    for i, summary in enumerate(summaries):
        records = [json.loads(line) for line in (args.output/f'checkpoint_{i}/tokens.jsonl').read_text().splitlines()]
        delta = np.array([a-b for row in records for a,b in zip(row['train_logp'], row['rollout_logp'])])
        axes[0].hist(delta, bins=np.linspace(-.15,.15,101), histtype='step', density=True, label=labels[i])
        positions = np.array([j for row in records for j in range(len(row['suffix']))])
        centers, errors = [], []
        for start in range(0,1024,128):
            mask = (positions >= start) & (positions < start+128)
            if mask.any():
                centers.append(start+64)
                errors.append(np.abs(delta[mask]).mean())
        axes[1].plot(centers, errors, marker='o', label=labels[i])
        sequence = np.array([sum(row['train_logp'])-sum(row['rollout_logp']) for row in records])
        axes[2].hist(sequence, bins=20, histtype='step', label=labels[i])
        m, g = summary['train_vs_vllm'], summary['gradient_comparison']
        table.append([labels[i], summary['tokens'], m['absolute_logprob_error']['mean'],
                      m['token_probability_ratio']['p1'], m['token_probability_ratio']['p99'],
                      m['fraction_outside_08_12'], summary['top1_agreement'],
                      g['relative_l2_difference'], g['cosine']])
        checkpoint = Path(summary['checkpoint'])
        metadata = checkpoint.with_name(checkpoint.name.replace('model_', 'meta_')).with_suffix('.json')
        manifest.append(dict(checkpoint=str(checkpoint), checkpoint_sha256=summary['checkpoint_sha256'],
                             metadata=str(metadata), metadata_sha256=sha256(metadata)))
        cached = [row for row in records if 'native_cached_logp' in row]
        def error(left, right):
            return float(np.mean([abs(a-b) for row in cached for a,b in zip(row[left],row[right])]))
        controls.append(dict(checkpoint=labels[i], responses=len(cached),
            tokens=sum(len(row['suffix']) for row in cached),
            train_vs_vllm_mae=error('train_logp','rollout_logp'),
            train_vs_native_cached_mae=error('train_logp','native_cached_logp'),
            native_cached_vs_vllm_mae=error('native_cached_logp','rollout_logp')))
    axes[0].set(xlabel='Token log(p_train / p_vLLM)', ylabel='Density (display range ±0.15)', yscale='log')
    axes[1].set(xlabel='Response token position', ylabel='Mean absolute log-probability error')
    axes[2].set(xlabel='Sequence log(p_train / p_vLLM)', ylabel='Response count')
    for axis in axes:
        axis.axvline(0, color='gray', linestyle=':', linewidth=1)
        axis.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(f"Train/inference consistency · {summaries[0]['gpu']} · synchronized weights")
    plot = args.output/'train_inference.png'
    fig.savefig(plot, dpi=180)
    (args.output/'audit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'control_comparison.json').write_text(json.dumps(controls, indent=2)+'\n')
    if args.wandb:
        import wandb
        run = wandb.init(project='nanochat-online-rl', name=f'{args.output.name}-sanity',
                         job_type='train-inference-sanity', dir=str(args.output),
                         save_code=False, settings=wandb.Settings(disable_git=True, console='off'),
                         config={'diagnostic':'train-inference consistency',
                                 'prompts_per_checkpoint':summaries[0]['prompts'],
                                 'responses_per_checkpoint':summaries[0]['responses'],
                                 'gpu':summaries[0]['gpu']})
        for i, summary in enumerate(summaries):
            run.log(dict(checkpoint_index=i, **flatten(summary), **flatten(controls[i], 'matched_control')))
        run.log({'audit/comparison':wandb.Table(columns=[
            'checkpoint','tokens','mean_abs_logprob_error','ratio_p1','ratio_p99',
            'fraction_ratio_outside_08_12','top1_agreement','gradient_relative_l2','gradient_cosine'],data=table)})
        (args.output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url),indent=2)+'\n')
        run.finish()
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
