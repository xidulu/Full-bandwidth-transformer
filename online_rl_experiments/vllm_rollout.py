"""Synchronous, colocated vLLM rollouts isolated from the trainer's DDP group.

One subprocess per training rank owns a single-GPU vLLM engine. A persistent
shared CPU state dict transfers current weights without checkpoint I/O or CUDA
IPC lifetimes. The parent cannot update it until the worker acknowledges loading.
"""
import atexit
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.multiprocessing as mp


def inference_weights(model, decode_mode='standard'):
    """Match the adapter loader, including feedback weights in soft mode."""
    if decode_mode not in ('standard', 'soft'):
        raise ValueError(f'Unsupported decode mode: {decode_mode}')
    vocab = model.config.vocab_size
    for name, tensor in model.named_parameters():
        if name.startswith('latent_feedback.') and decode_mode == 'standard':
            continue
        if name == 'lm_head.weight' and model.config.weight_tying:
            continue
        if name in ('transformer.wte.weight', 'lm_head.weight') or name.startswith('value_embeds.'):
            tensor = tensor[:vocab]
        yield name, tensor.detach()


def prepare_model_config(model_config, output, decode_mode='standard'):
    from nanochat_vllm.export_checkpoint import _hf_config
    if decode_mode not in ('standard', 'soft'):
        raise ValueError(f'Unsupported decode mode: {decode_mode}')
    if decode_mode == 'soft' and not model_config.get('latent_feedback'):
        raise ValueError('Soft rollouts require latent-feedback weights')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(_hf_config(model_config, decode_mode), indent=2) + '\n')


def visible_device(local_rank, visible=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    devices = visible.split(',') if visible is not None else None
    return devices[local_rank].strip() if devices is not None else str(local_rank)


def soft_sequence_capacity(config, kv_cache_gb):
    """Conservative full-context KV capacity; recurrent state cannot be replayed.

The installed soft runner refuses preemption replay across prompt/response
boundaries. Reserve enough BF16 KV space for every active request at max length.
    """
    bytes_per_token = 2 * config.n_layer * config.n_kv_head * (config.n_embd // config.n_head) * 2
    padded_context = ((config.sequence_len + 15) // 16) * 16
    capacity = int(0.9 * kv_cache_gb * 1024**3 // (bytes_per_token * padded_context))
    if capacity < 1:
        raise ValueError('KV budget cannot safely hold one full-context soft request')
    return capacity


def _load_policy(worker, state, version, verify):
    model = worker.model_runner.model
    loaded = model.load_weights(state.items())
    expected = set(dict(model.named_parameters()))
    if loaded != expected:
        raise RuntimeError(f'Incomplete vLLM weight load: missing={expected - loaded}, extra={loaded - expected}')
    if verify:
        for name, parameter in model.named_parameters():
            source = state[name].to(device=parameter.device, dtype=parameter.dtype)
            # vLLM may add vocabulary padding internally.
            actual = parameter[:source.shape[0]] if parameter.ndim else parameter
            if not torch.equal(actual, source):
                raise RuntimeError(f'vLLM weight verification failed: {name}')
    torch.cuda.synchronize()
    return {'policy_version': version, 'loaded_parameters': len(loaded), 'weights_verified': verify}


def unpack_outputs(outputs, count, stops, max_tokens):
    if len(outputs) != count:
        raise RuntimeError('vLLM returned the wrong number of requests')
    suffixes, ended = [], []
    for request in outputs:
        if not request.finished or len(request.outputs) != 1:
            raise RuntimeError('vLLM returned an incomplete request')
        completion = request.outputs[0]
        tokens = list(completion.token_ids)
        if not tokens or len(tokens) > max_tokens:
            raise RuntimeError('Invalid vLLM completion length')
        terminal = tokens[-1] in stops
        if any(token in stops for token in tokens[:-1]):
            raise RuntimeError('vLLM returned tokens after a terminal token')
        if completion.finish_reason not in ('stop', 'length'):
            raise RuntimeError(f'Unexpected vLLM finish reason: {completion.finish_reason}')
        if completion.finish_reason == 'stop' and not terminal:
            raise RuntimeError('vLLM omitted the sampled terminal token')
        # A terminal sampled at max_tokens is completed, just like native Engine.
        suffixes.append(tokens)
        ended.append(terminal)
    return suffixes, ended


def _worker(connection, state, settings):
    llm = None
    try:
        # vLLM must not discover or join the parent's torchrun process group.
        for key in list(os.environ):
            if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                       'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK',
                       'ROLE_WORLD_SIZE') or key.startswith('TORCHELASTIC_'):
                os.environ.pop(key)
        os.environ['CUDA_VISIBLE_DEVICES'] = settings['device']
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        os.environ.pop('VLLM_USE_V2_MODEL_RUNNER', None)
        from nanochat_vllm import register
        register()
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=settings['model_dir'], skip_tokenizer_init=True,
            worker_cls=('replay_worker.ReplayWorker' if settings['hidden_replay'] else
                        'nanochat_vllm.worker.NanochatWorker'),
            distributed_executor_backend='uni', tensor_parallel_size=1,
            enforce_eager=True, dtype='bfloat16', load_format='dummy',
            enable_prefix_caching=False, async_scheduling=False,
            max_model_len=settings['context'], max_num_seqs=settings['max_sequences'],
            max_num_batched_tokens=settings['max_batched_tokens'],
            kv_cache_memory_bytes=settings['kv_cache_bytes'],
            gpu_memory_utilization=0.2, seed=settings['seed'],
            disable_log_stats=True,
        )
        version = None
        connection.send(('ok', None))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command == 'sync':
                result, = llm.collective_rpc(_load_policy, args=(state, payload, settings['verify_weights']))
                version = payload
            elif command in ('generate', 'generate_scored'):
                if version is None or version != payload['policy_version']:
                    raise RuntimeError('Refusing to generate from an unsynchronized policy')
                params = [SamplingParams(
                    n=1, temperature=payload['temperature'], top_p=1.0, top_k=-1,
                    max_tokens=payload['max_tokens'], seed=seed,
                    stop_token_ids=settings['stops'], ignore_eos=True, detokenize=False,
                    logprobs=1 if command == 'generate_scored' else None,
                ) for seed in payload['seeds']]
                if payload.get('return_hidden_states'):
                    from replay_worker import begin_replay
                    llm.collective_rpc(begin_replay)
                from hidden_replay import capture_request_ids
                capture = (capture_request_ids(llm.llm_engine.input_processor)
                           if payload.get('return_hidden_states') else nullcontext())
                with capture as request_ids:
                    outputs = llm.generate(
                        [{'prompt_token_ids': prompt} for prompt in payload['prompts']],
                        params, use_tqdm=False)
                result = unpack_outputs(outputs, len(params), settings['stops'], payload['max_tokens'])
                if command == 'generate_scored':
                    suffixes, ended = result
                    scores = []
                    for request, tokens in zip(outputs, suffixes):
                        logs = request.outputs[0].logprobs
                        if logs is None or len(logs) != len(tokens):
                            raise RuntimeError('Missing rollout token log-probabilities')
                        scores.append(dict(
                            logprobs=[row[token].logprob for token, row in zip(tokens, logs)],
                            top1=[max(row, key=lambda token: row[token].logprob) for row in logs]))
                    if payload.get('return_hidden_states'):
                        from replay_worker import finish_replay
                        requests = [(request_ids[request.request_id], len(prompt), len(tokens))
                                    for request, prompt, tokens in zip(outputs, payload['prompts'], suffixes)]
                        states, = llm.collective_rpc(finish_replay, args=(requests,))
                        for score, hidden in zip(scores, states):
                            score['hidden_states'] = hidden
                    result = (suffixes, ended, scores)
            else:
                raise ValueError(f'Unknown rollout command: {command}')
            connection.send(('ok', result))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        if torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory()


class VLLMRollout:
    def __init__(self, model, tokenizer, model_dir, local_rank, max_sequences,
                 kv_cache_gb=4.0, max_batched_tokens=2048, seed=42, verify_weights=False,
                 decode_mode='standard', hidden_replay=False):
        if hidden_replay and decode_mode != 'soft':
            raise ValueError('Hidden replay requires soft decoding')
        self.hidden_replay = hidden_replay
        if importlib.metadata.version('vllm') != '0.14.0':
            raise RuntimeError('This rollout adapter requires vLLM==0.14.0')
        config = json.loads((Path(model_dir) / 'config.json').read_text())
        if config['nanochat_decode_mode'] != decode_mode:
            raise ValueError('vLLM export decode mode differs from rollout decode mode')
        self.decode_mode = decode_mode
        self.max_sequences = (min(max_sequences, soft_sequence_capacity(model.config, kv_cache_gb))
                              if decode_mode == 'soft' else max_sequences)
        self.state = {name: torch.empty(tensor.shape, dtype=tensor.dtype).share_memory_()
                      for name, tensor in inference_weights(model, decode_mode)}
        self.policy_version = None
        self.timeout = 1800
        settings = dict(
            model_dir=str(Path(model_dir).resolve()), device=visible_device(local_rank),
            context=model.config.sequence_len, max_sequences=self.max_sequences,
            kv_cache_bytes=int(kv_cache_gb * 1024**3), max_batched_tokens=max_batched_tokens,
            seed=seed, verify_weights=verify_weights, hidden_replay=hidden_replay,
            stops=[tokenizer.get_bos_token_id(), tokenizer.encode_special('<|assistant_end|>')])
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, self.state, settings), daemon=True)
        self.process.start()
        child.close()
        atexit.register(self.close)
        self._receive()

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f'vLLM worker exited with code {self.process.exitcode}')
            if time.monotonic() > deadline:
                raise TimeoutError('vLLM worker response timed out')
        try:
            status, result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError('vLLM worker disconnected') from error
        if status != 'ok':
            raise RuntimeError(f'vLLM worker failed:\n{result}')
        return result

    def sync_weights(self, model, policy_version):
        if self.policy_version == policy_version:
            return 0.0
        started = time.perf_counter()
        for name, tensor in inference_weights(model, self.decode_mode):
            self.state[name].copy_(tensor)
        self.connection.send(('sync', policy_version))
        result = self._receive()
        if result['policy_version'] != policy_version:
            raise RuntimeError('vLLM acknowledged the wrong policy version')
        self.policy_version = policy_version
        return time.perf_counter() - started

    def generate_groups(self, prompts, seeds, n, max_tokens, policy_version, temperature=1.0,
                        return_scores=False, return_hidden_states=False):
        if return_hidden_states and not getattr(self, 'hidden_replay', False):
            raise ValueError('Enable hidden_replay when constructing the rollout engine')
        return_scores = return_scores or return_hidden_states
        if self.policy_version is None or self.policy_version != policy_version:
            raise RuntimeError('Rollout policy is stale; synchronize before generating')
        if len(prompts) != len(seeds) or not prompts:
            raise ValueError('Expected one seed per prompt')
        command = 'generate_scored' if return_scores else 'generate'
        self.connection.send((command, dict(
            prompts=[prompt for prompt in prompts for _ in range(n)],
            seeds=[seed + i for seed in seeds for i in range(n)],
            max_tokens=max_tokens, temperature=temperature, policy_version=policy_version,
            return_hidden_states=return_hidden_states)))
        result = self._receive()
        if return_scores:
            suffixes, ended, scores = result
            return [(suffixes[i:i+n], ended[i:i+n], scores[i:i+n]) for i in range(0, len(suffixes), n)]
        suffixes, ended = result
        return [(suffixes[i:i+n], ended[i:i+n]) for i in range(0, len(suffixes), n)]

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()
