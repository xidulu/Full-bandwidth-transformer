"""Use the repository's full shard validation with a soft-only-safe report."""
from fbt_experiments import merge_gsm8k_shards as merger
from evaluate_rl_gsm8k import render_gsm8k_summary


if __name__ == '__main__':
    merger.render_summary = render_gsm8k_summary
    merger.main()
