#!/usr/bin/env python3
"""Create a concise report and standalone plots from replay_experiment results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    folder = args.result_dir
    result = json.loads((folder/'results.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    aggregate = result['aggregate']
    labels = [label for label in aggregate if label.startswith('pass_')]
    passes = [int(label.split('_')[1]) for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(6, 1.2), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    for key, color in [('k', '#1565c0'), ('v', '#c94c15')]:
        axes[0].plot(passes, [aggregate[label][key]['relative_l2'] for label in labels],
                     marker='o', markersize=2.5, linewidth=1, color=color, label=key.upper())
        axes[0].axhline(aggregate['oracle_hidden'][key]['relative_l2'],
                        color=color, linestyle=':', linewidth=1, label=f'{key.upper()} (control)')
    axes[0].set(yscale='log')
    axes[0].set_ylabel(r'Relative $\ell_2$ error', fontsize=7, labelpad=2)
    axes[0].legend(fontsize=5.5, frameon=False, ncol=2, loc='upper right',
                   handlelength=1.5, columnspacing=0.8, borderaxespad=0.2)
    axes[1].plot(passes, [aggregate[label]['logits']['top1_agreement'] for label in labels],
                 marker='o', markersize=2.5, linewidth=1, color='#357a38')
    axes[1].axhline(aggregate['oracle_hidden']['logits']['top1_agreement'],
                    color='gray', linestyle=':', linewidth=1, label='Recorded-hidden control')
    axes[1].set(ylim=(0.9, 1.005), yticks=[0.9, 0.95, 1.0])
    axes[1].set_ylabel('Argmax\nagreement', fontsize=7, labelpad=2)
    axes[1].legend(fontsize=5.5, frameon=False, loc='lower right', borderaxespad=0.2)
    for ax in axes:
        ax.set(xscale='log')
        ax.set_xlabel('Total forward passes', fontsize=7, labelpad=2)
        ax.set_xticks(passes, [str(p) for p in passes])
        ax.minorticks_off()
        ax.tick_params(labelsize=6, length=2, width=0.5, pad=2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(0.5)
        ax.grid(axis='y', alpha=0.15, linewidth=0.5)
    fig.savefig(folder/'convergence.png', dpi=300)
    fig.savefig(folder/'convergence.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(len(result['records']), 2, figsize=(11, 2.6*len(result['records'])),
                             squeeze=False, constrained_layout=True)
    for row, record in enumerate(result['records']):
        for col, key in enumerate(('k', 'v')):
            ax = axes[row, col]
            for label in ('pass_01', 'pass_03', 'pass_08', 'pass_32', 'oracle_hidden'):
                if label not in record['comparisons']:
                    continue
                values = record['comparisons'][label][key].get('response', {}).get('relative_l2_by_position', [])
                if values:
                    ax.plot(np.arange(1, len(values)+1), values, label=label, linewidth=0.9)
            ax.set(title=f'Example {record["index"]}: {key.upper()}', xlabel='Consumed response-token position',
                   ylabel='Relative L2 error', yscale='log')
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(folder/'position_errors.png', dpi=160)
    plt.close(fig)

    lines = [f'# MATH-500 soft-decoding KV reconstruction', '',
             f'Checkpoint: `{manifest["checkpoint"]}` (step {manifest["checkpoint_step"]}).', '',
             f'{len(result["records"])} greedy rollouts; {manifest["compute_dtype"]}; '
             f'{manifest["attention_backend"]}; {manifest["gpu"]}.', '',
             '| Passes | K relative L2 | V relative L2 | K cosine | V cosine | KL (nats) | Argmax agreement |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels + ['oracle_hidden']:
        value = aggregate[label]
        name = str(int(label.split('_')[1])) if label.startswith('pass_') else 'Recorded-hidden control'
        lines.append(f'| {name} | {value["k"]["relative_l2"]:.5f} | {value["v"]["relative_l2"]:.5f} '
                     f'| {value["k"]["cosine"]:.5f} | {value["v"]["cosine"]:.5f} '
                     f'| {value["logits"]["kl_true_to_replay_mean"]:.5f} '
                     f'| {100*value["logits"]["top1_agreement"]:.2f}% |')
    lines += ['', 'Cache metrics pool response inputs only across layers and examples. '
              'Relative L2 = ||replay − true||₂ / ||true||₂. '
              'Logit metrics include every sampled token, including the first and terminal token when present.', '',
              'The recorded-hidden control measures one full parallel pass supplied with the actual '
              'recurrent hidden states. Its remaining error measures BF16/backend numerical differences.', '',
              '| Example | Prompt tokens | Sampled tokens | Stop |', '|---|---:|---:|---|']
    for r in result['records']:
        lines.append(f'| {r["index"]} | {r["prompt_tokens"]} | {r["generated_tokens"]} | {r["stop_reason"]} |')
    lines += ['', '## Alignment and interpretation', '',
              '- Cache inputs are `prompt + generated[:-1]`. The last sampled token is a target and has not been consumed.',
              '- Pass 1 uses ordinary inputs. Later passes shift the previous hidden states by one token and apply feedback only to response inputs.',
              '- Each pass starts with an empty cache at position zero. Prompt positions remain ordinary.',
              '- In exact arithmetic, pass K recovers the first K−1 response-input positions; later positions need not be accurate at finite K.',
              '- This is a four-example cache diagnostic, not a MATH-500 accuracy estimate. No correctness score is reported.', '',
              '![Convergence](convergence.png)', '', '![Errors by response position](position_errors.png)', '']
    (folder/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()
