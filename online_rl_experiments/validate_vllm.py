"""GPU acceptance check: batched greedy parity and a changed-weight round trip."""
import argparse
import json
from pathlib import Path

import torch

from main import TAG, collect
from vllm_rollout import VLLMRollout, prepare_model_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from nanochat.checkpoint_manager import build_model
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.engine import Engine
    model, tokenizer, _ = build_model(
        f'/home/jhu/xwang457/work/nanochat_cache/chatsft_checkpoints/{TAG}',
        4407, torch.device('cuda'), 'train')
    model.float().eval()
    model.tie_weights()
    model.cos = model.cos.to(COMPUTE_DTYPE)
    model.sin = model.sin.to(COMPUTE_DTYPE)
    native = Engine(model, tokenizer)
    texts = ['What is 17 times 23?', 'What is 2 + 2?',
             'A box has 12 apples. Three children each take 2 apples. How many remain?',
             'Compute 125 divided by 5.']
    prompts = [tokenizer.render_for_completion({'messages': [
        {'role': 'user', 'content': text}, {'role': 'assistant', 'content': ''}]}) for text in texts]
    expected = [collect(native, prompt, 1, 32, 42, temperature=0.0) for prompt in prompts]
    prepare_model_config(vars(model.config), args.output / 'model')
    engine = VLLMRollout(model, tokenizer, args.output / 'model', 0, 32, verify_weights=True)
    try:
        engine.sync_weights(model, 0)
        actual = engine.generate_groups(prompts, [42] * len(prompts), 1, 32, 0, temperature=0.0)
        records = [dict(prompt=text, native=left, vllm=right, match=left == right)
                   for text, left, right in zip(texts, expected, actual)]
        # Exercise an actual changed policy, including the FP32 residual scalars.
        original = model.resid_lambdas.detach().clone()
        with torch.no_grad():
            model.resid_lambdas.add_(0.1)
        engine.sync_weights(model, 1)  # audits every tensor against the changed policy
        with torch.no_grad():
            model.resid_lambdas.copy_(original)
        engine.sync_weights(model, 2)
        restored = engine.generate_groups(prompts, [42] * len(prompts), 1, 32, 2, temperature=0.0)
        report = dict(records=records, greedy_match=all(row['match'] for row in records),
                      changed_weights_verified=True, restored_match=restored == actual)
        (args.output / 'parity.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        assert report['greedy_match'] and report['restored_match'], 'vLLM acceptance check failed'
    finally:
        engine.close()


if __name__ == '__main__':
    main()
