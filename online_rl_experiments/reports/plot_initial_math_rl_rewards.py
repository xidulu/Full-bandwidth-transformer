"""Plot saved initial-RL rewards; --extract refreshes the compact source CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUNS = [
    ('standard', 'Standard', '899298', '899298-preflight', '#1764ab'),
    ('three_pass', 'LF three-pass', '906530', '904517-preflight', '#d27b13'),
    ('hidden_replay', 'LF hidden-state replay', '906531', '904906-preflight', '#14836b'),
]
WINDOW = 25


def extract():
    records, sources = [], []
    for key, label, job, preflight, _ in RUNS:
        for segment in [preflight, job]:
            directory = ROOT / 'results' / segment
            config = json.loads((directory / 'run_config.json').read_text())
            assert config['prompts_per_step'] == 128
            assert config['samples_per_prompt'] == 8
            assert config['lr'] == 1e-6 and config['max_new_tokens'] == 1024
            assert config['temperature'] == 1.0
            sources.append({
                'run': key, 'segment': segment,
                'metrics_sha256': hashlib.sha256((directory / 'metrics.jsonl').read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256((directory / 'run_config.json').read_bytes()).hexdigest(),
                'data_sha256': config['data_sha256'],
            })
            for row in map(json.loads, (directory / 'metrics.jsonl').read_text().splitlines()):
                if 'reward/mean' in row:
                    assert row['batch/global_sequences'] == 1024
                    records.append({'run': key, 'job_id': job, 'segment': segment,
                                    'step': row['step'], 'reward_mean': row['reward/mean']})
    assert len({s['data_sha256'] for s in sources}) == 1
    with (OUT / 'initial_math_rl_rewards.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'job_id', 'segment', 'step', 'reward_mean'],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (OUT / 'initial_math_rl_reward_sources.json').write_text(json.dumps(sources, indent=2) + '\n')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    with (OUT / 'initial_math_rl_rewards.csv').open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 900
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'initial-math-rl'})
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=.09, right=.97, bottom=.25, top=.77)
    fig.text(.09, .94, 'Reward improves during initial math RL', fontsize=19, weight='bold')
    fig.text(.09, .89, 'Hendrycks-MATH training split  |  128 questions × 8 responses per update', color='#505861')
    stats = {}
    for i, (key, label, job, _, color) in enumerate(RUNS):
        rows = [r for r in records if r['run'] == key]
        steps = [int(r['step']) for r in rows]
        values = [float(r['reward_mean']) for r in rows]
        assert steps == list(range(1, 301))
        assert all(0 <= v <= 1 and (v * 1024).is_integer() for v in values)
        smooth = [mean(values[j-WINDOW+1:j+1]) for j in range(WINDOW-1, len(values))]
        ax.plot(steps, values, color=color, alpha=.16, lw=.8)
        ax.plot(steps[WINDOW-1:], smooth, color=color, lw=2.5,
                linestyle='--' if key == 'hidden_replay' else '-', label=label)
        first, last = mean(values[:WINDOW]), mean(values[-WINDOW:])
        stats[key] = {'job_id': job, 'first_25_mean': first, 'last_25_mean': last,
                      'gain_percentage_points': 100 * (last-first)}
        x = .09 + i * .305
        fig.text(x, .135, label, color=color, weight='bold')
        fig.text(x, .095, f'{first:.1%} → {last:.1%}  (+{100*(last-first):.1f} pp)', fontsize=13)
    ax.set(xlim=(1, 300), ylim=(.30, .65), xlabel='RL update', ylabel='Mean training reward')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks([1, 50, 100, 150, 200, 250, 300])
    ax.grid(axis='y', color='#dde2e7', lw=.7)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#bbc2ca')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, borderaxespad=0)
    fig.text(.09, .035, 'Faint: individual updates. Bold: trailing 25-update mean. Below: first vs last 25 updates.',
             fontsize=10, color='#505861')
    fig.savefig(OUT / 'initial_math_rl_rewards.png', dpi=180, facecolor='white')
    fig.savefig(OUT / 'initial_math_rl_rewards.svg', facecolor='white', metadata={'Date': None})
    svg = OUT / 'initial_math_rl_rewards.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    (OUT / 'initial_math_rl_reward_summary.json').write_text(json.dumps({
        'metric': 'reward/mean', 'smoothing': 'trailing 25 updates, full windows only',
        'dataset': 'nlile/hendrycks-MATH-benchmark', 'split': 'train',
        'max_new_tokens': 1024, 'temperature': 1.0, 'lr': 1e-6,
        'reward': 'Binary Math-Verify correctness; truncated training responses receive zero.',
        'runs': stats,
    }, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    if args.extract:
        extract()
    plot()
