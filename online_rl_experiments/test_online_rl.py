import torch
import pytest
from main import collect, group_advantages, pack, token_loss



def test_group_signal():
    assert torch.equal(group_advantages([1, 1, 1]), torch.zeros(3))
    assert torch.equal(group_advantages([0, 0, 0]), torch.zeros(3))
    torch.testing.assert_close(group_advantages([0, 1]), torch.tensor([-1., 1.]))


def test_global_token_normalization_gradient_is_microbatch_invariant():
    # Different lengths and advantages expose accidental per-sequence/microbatch averaging.
    nll = torch.tensor([[2., 4., 9.], [1., 3., 5.]], requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    adv = torch.tensor([-1., 1.])
    whole = token_loss(nll, adv, mask, 4)
    expected = (-2 + 1 + 3 + 5) / 4
    assert whole.item() == expected
    whole.backward()
    grad = nll.grad.clone()
    nll.grad = None
    for i in range(2):
        token_loss(nll[i:i+1], adv[i:i+1], mask[i:i+1], 4).backward()
    torch.testing.assert_close(nll.grad, grad)
    assert grad[0, 0] < 0 < grad[1, 0]  # gradient descent lowers NLL on rewarded actions
    assert grad[0, 1] == 0


def test_pack_masks_prompt_padding_and_includes_eos():
    samples = [dict(prompt=[1, 2], suffix=[3, 9], advantage=1),
               dict(prompt=[1, 2, 4], suffix=[9], advantage=-1)]
    inputs, targets, adv = pack(samples, 9, 'cpu')
    assert targets.tolist() == [[-1, 3, 9], [-1, -1, 9]]
    assert inputs.tolist() == [[1, 2, 3], [1, 2, 4]]
    assert adv.tolist() == [1, -1]


def test_collect_retains_terminal_and_excludes_post_terminal_tokens():
    class Tokenizer:
        def get_bos_token_id(self): return 0
        def encode_special(self, name): return 9
    class Engine:
        tokenizer = Tokenizer()
        closed = False
        def generate(self, prompt, **kwargs):
            assert kwargs['temperature'] == 1.0 and kwargs['top_k'] is None
            assert kwargs['use_calculator'] is False
            try:
                yield [9, 5], [1, 1]
                yield [6, 7], [1, 1]
                yield [8, 9], [1, 1]
            finally:
                self.closed = True
    engine = Engine()
    seq, ended = collect(engine, [1], 2, 3, 42)
    assert seq == [[9], [5, 7, 9]]
    assert ended == [True, True]
    assert engine.closed


def test_four_gpu_batch_layout():
    from main import batch_layout
    assert batch_layout(8, 4, 4, 8) == (16, 32)
    with pytest.raises(ValueError, match='must equal 16'):
        batch_layout(8, 4, 4, 8, prompts_per_step=4)
    with pytest.raises(ValueError, match='complete response groups'):
        batch_layout(3, 4, 4, 8)


def _distributed_gradient_worker(rank, world_size, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from main import reduce_values, sync_context
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=world_size)
    try:
        torch.manual_seed(123)
        initial = torch.nn.Linear(2, 1, bias=False).double()
        model = DDP(initial)
        generator = torch.Generator().manual_seed(99)
        # Unequal valid-token counts across ranks/microbatches; rank 0 has no signal.
        x = torch.randn(world_size, 8, 5, 2, generator=generator, dtype=torch.float64)
        mask = torch.arange(5)[None, None, :] < (torch.arange(world_size * 8).reshape(world_size, 8, 1) % 5 + 1)
        adv = torch.tensor([[0.] * 8, [-1., 1.] * 4, [1., -1.] * 4, [-1., 1.] * 4], dtype=torch.float64)
        denominator = reduce_values([mask[rank].sum().item()], 'cpu')[0]
        assert denominator == mask.sum().item()
        for micro in range(4):
            sl = slice(micro * 2, micro * 2 + 2)
            with sync_context(model, micro, 4):
                nll = model(x[rank, sl]).squeeze(-1).square()
                loss = token_loss(nll, adv[rank, sl], mask[rank, sl], denominator) * world_size
                loss.backward()
        reference = torch.nn.Linear(2, 1, bias=False).double()
        reference.load_state_dict(initial.state_dict())
        all_nll = reference(x).squeeze(-1).square()
        ((all_nll * adv[..., None] * mask).sum() / mask.sum()).backward()
        torch.testing.assert_close(initial.weight.grad, reference.weight.grad, rtol=1e-10, atol=1e-10)
        optimizer = torch.optim.SGD(initial.parameters(), lr=0.01)
        optimizer.step()
        gathered = [torch.zeros_like(initial.weight) for _ in range(world_size)]
        dist.all_gather(gathered, initial.weight.detach())
        for weight in gathered:
            torch.testing.assert_close(weight, reference.weight.detach() - 0.01 * reference.weight.grad,
                                       rtol=1e-10, atol=1e-10)
    finally:
        dist.destroy_process_group()


def test_four_process_accumulated_gradient_matches_global_token_mean(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_gradient_worker, args=(4, str(tmp_path / 'gloo_init')), nprocs=4, join=True)


@pytest.mark.parametrize('total, successful, mixed, expected', [
    (16, 0, 0, (16, 0, 0)),
    (16, 16, 0, (0, 16, 0)),
    (16, 16, 16, (0, 0, 16)),
    (16, 6, 4, (10, 2, 4)),
])
def test_global_group_metrics_partition_batch(total, successful, mixed, expected):
    from main import group_outcome_metrics
    metrics = group_outcome_metrics(total, successful, mixed)
    names = ('all_wrong', 'all_correct', 'mixed')
    assert tuple(metrics[f'reward/{name}_groups'] for name in names) == expected
    assert sum(metrics[f'reward/{name}_groups'] for name in names) == total
    assert sum(metrics[f'reward/{name}_group_fraction'] for name in names) == 1.0


@pytest.mark.parametrize('consumed', [0, 7, 8, 23])
def test_resumed_shuffle_matches_uninterrupted_stream(consumed):
    from main import restore_data_order
    size, seed = 7, 42
    rng, order, cursor, epoch = restore_data_order(size, seed)
    def next_item(rng, order, cursor, epoch):
        if cursor == len(order):
            rng.shuffle(order)
            cursor, epoch = 0, epoch + 1
        return order[cursor], cursor + 1, epoch
    for _ in range(consumed):
        _, cursor, epoch = next_item(rng, order, cursor, epoch)
    resumed_rng, resumed_order, resumed_cursor, resumed_epoch = restore_data_order(size, seed, epoch, cursor)
    for _ in range(20):
        expected, cursor, epoch = next_item(rng, order, cursor, epoch)
        actual, resumed_cursor, resumed_epoch = next_item(resumed_rng, resumed_order, resumed_cursor, resumed_epoch)
        assert (actual, resumed_cursor, resumed_epoch) == (expected, cursor, epoch)


def test_optimizer_resume_preserves_next_update(tmp_path):
    from main import restore_optimizer
    p = torch.nn.Parameter(torch.tensor([0.5, -0.7]))
    original = torch.optim.AdamW([p], lr=1e-6, betas=(0.9, 0.95), weight_decay=0)
    p.grad = torch.tensor([0.3, -0.2])
    original.step()
    path = tmp_path / 'optim.pt'
    torch.save(original.state_dict(), path)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = torch.optim.AdamW([q], lr=1e-6, betas=(0.9, 0.95), weight_decay=0)
    restore_optimizer(resumed, path)
    p.grad = q.grad = torch.tensor([-0.2, 0.4])
    original.step()
    resumed.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(original.state[p][key], resumed.state[q][key], rtol=0, atol=0)


def test_resume_rejects_changed_batch_or_data():
    from main import validate_resume_config
    saved = dict(data_sha256='abc', world_size=4, steps=100)
    validate_resume_config(saved, dict(saved, steps=300))
    with pytest.raises(ValueError, match='world_size'):
        validate_resume_config(saved, dict(saved, world_size=1))
    with pytest.raises(ValueError, match='data_sha256'):
        validate_resume_config(saved, dict(saved, data_sha256='different'))


def test_learning_rate_branch_requires_explicit_override():
    from main import validate_resume_config
    saved = dict(lr=1e-6, seed=42, grad_clip=1., world_size=4)
    with pytest.raises(ValueError, match='lr'):
        validate_resume_config(saved, dict(saved, lr=1e-5))
    changed = dict(saved, lr=1e-5, allow_learning_rate_change=True)
    validate_resume_config(saved, changed)
    for key, value in [('seed', 43), ('grad_clip', 2.), ('world_size', 1)]:
        with pytest.raises(ValueError, match=key):
            validate_resume_config(saved, dict(changed, **{key:value}))


def test_learning_rate_override_preserves_moments_and_scales_next_update(tmp_path):
    from main import restore_optimizer
    params = [torch.nn.Parameter(torch.tensor([v], dtype=torch.float64)) for v in [.5, -.7]]
    original = torch.optim.AdamW([{'params':[p]} for p in params], lr=1e-6,
                                 betas=(.9,.95), weight_decay=0)
    for p in params:
        p.grad = torch.tensor([.3], dtype=torch.float64)
    original.step()
    path = tmp_path/'optim.pt'
    torch.save(original.state_dict(), path)
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = torch.optim.AdamW([{'params':[p]} for p in copies], lr=1e-5)
    restore_optimizer(resumed, path, learning_rate=1e-5)
    assert all(g['lr']==1e-5 and g['betas']==(.9,.95) and g['weight_decay']==0 for g in resumed.param_groups)
    before = [p.detach().clone() for p in params]
    for p,q in zip(params,copies):
        for key in ['step','exp_avg','exp_avg_sq']:
            torch.testing.assert_close(original.state[p][key],resumed.state[q][key],rtol=0,atol=0)
        p.grad = q.grad = torch.tensor([-.2],dtype=torch.float64)
    original.step()
    resumed.step()
    for p,q,b in zip(params,copies,before):
        torch.testing.assert_close(q-b,10*(p-b),rtol=1e-8,atol=1e-15)
        for key in ['step','exp_avg','exp_avg_sq']:
            torch.testing.assert_close(original.state[p][key],resumed.state[q][key],rtol=0,atol=0)
    # A subsequent ordinary resume retains the new rate without another override.
    next_path = tmp_path/'optim_next.pt'
    torch.save(resumed.state_dict(),next_path)
    restored = torch.optim.AdamW([{'params':[p]} for p in copies],lr=1e-6)
    restore_optimizer(restored,next_path)
    assert all(g['lr']==1e-5 for g in restored.param_groups)
