"""Run the repository GSM8K evaluator on an exact external RL checkpoint."""
from fbt_experiments import evaluate_checkpoint as evaluator
from evaluate_rl_math500 import checkpoint_paths


def render_gsm8k_summary(metrics, meta, args):
    """Render only the modes present; the upstream report assumes standard exists."""
    lines = ['# GSM8K evaluation', '', f'Checkpoint: `{args.checkpoint}`', '',
             'Native numeric grading; see metrics_math_verify.json for final Math-Verify scores.', '',
             '| Mode | Correct | Accuracy |', '|---|---:|---:|']
    for mode in metrics['modes']:
        row = metrics['gsm8k'][mode]
        lines.append(f"| {mode} | {row['correct']}/{row['examples']} | {row['accuracy']:.2%} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    evaluator.checkpoint_paths = checkpoint_paths
    evaluator.render_summary = render_gsm8k_summary
    evaluator.main()
