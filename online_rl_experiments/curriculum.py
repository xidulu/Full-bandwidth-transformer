"""Deterministic source/difficulty curriculum with checkpointed online statistics."""
from collections import defaultdict
import hashlib
import random

from calibrate_big_math import difficulty


class CurriculumSampler:
    def __init__(self, data, spec, seed, start_step, state=None):
        self.spec, self.seed = spec, seed
        self.groups = defaultdict(list)
        self.index_keys = []
        for i, row in enumerate(data):
            key = row['source'] + '/' + difficulty(row['llama8b_solve_rate'])
            self.groups[key].append(i)
            self.index_keys.append(key)
        self.keys = sorted(self.groups)
        self.key_indices = {k:i for i,k in enumerate(self.keys)}
        if set(self.keys) != set(spec['strata']):
            raise ValueError('Curriculum strata differ from prepared calibration data')
        for key in self.keys:
            if len(self.groups[key]) != spec['strata'][key]['population']:
                raise ValueError(f'Curriculum population mismatch: {key}')
        self.origin_step = state['origin_step'] if state else start_step
        self.completed_step = state['completed_step'] if state else start_step
        if self.completed_step != start_step:
            raise ValueError('Curriculum checkpoint step mismatch')
        self.positions = {k:list(state['positions'][k]) if state else [0,0] for k in self.keys}
        self.stats = {k:list(state['stats'][k]) if state else [0.0]*6 for k in self.keys}
        self.frozen = state['frozen'] if state else None
        self.orders = {}
        self.pending_step = None
        self.last_pool_counts = {}

    def _order(self, key):
        epoch, cursor = self.positions[key]
        cached = self.orders.get(key)
        if cached is None or cached[0] != epoch:
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{key}:{epoch}'.encode()).digest()[:8], 'big')
            order = list(self.groups[key])
            random.Random(seed).shuffle(order)
            self.orders[key] = (epoch, order)
        return self.orders[key][1]

    def _draw_one(self, key, used):
        for _ in range(2*len(self.groups[key])+1):
            if self.positions[key][1] == len(self.groups[key]):
                self.positions[key] = [self.positions[key][0]+1, 0]
            order = self._order(key)
            index = order[self.positions[key][1]]
            self.positions[key][1] += 1
            if index not in used:
                return index
        raise RuntimeError('No distinct question left in stratum')

    def draw(self, step):
        if step != self.completed_step+1 or self.pending_step is not None:
            raise ValueError('Curriculum draws must follow completed updates')
        rng = random.Random(self.seed + step*1000003)
        used, used_by_key, selected = set(), defaultdict(int), []
        self.last_pool_counts = {}
        for pool in self.spec['pools']:
            keys = self.keys if pool['name']=='exploration' else pool['strata']
            for _ in range(pool['questions']):
                available = [k for k in keys if used_by_key[k] < len(self.groups[k])]
                if not available:
                    raise ValueError(f'Not enough distinct questions for {pool["name"]}')
                # Initially sample questions uniformly within each prescribed pool.
                # Phase two retains source quotas but emphasizes strata with mixed groups.
                weights = [(len(self.groups[k])-used_by_key[k]) *
                           (self.frozen[k] if self.frozen and pool['name']!='exploration' else 1.0)
                           for k in available]
                key = rng.choices(available, weights=weights, k=1)[0]
                index = self._draw_one(key, used)
                used.add(index); used_by_key[key] += 1; selected.append(index)
            self.last_pool_counts[pool['name']] = pool['questions']
        rng.shuffle(selected)
        self.pending_step = step
        return selected

    def observe(self, step, totals):
        """All-rank sums: groups, mixed, reward/group, wrong, correct, trunc/group."""
        if step != self.pending_step or len(totals) != len(self.keys):
            raise ValueError('Curriculum observations do not match pending batch')
        if sum(row[0] for row in totals) != sum(p['questions'] for p in self.spec['pools']):
            raise ValueError('Curriculum did not receive every prompt group')
        metrics = {'curriculum/phase': 2 if self.frozen else 1}
        for key, values in zip(self.keys, totals):
            self.stats[key] = [a+b for a,b in zip(self.stats[key],values)]
            if values[0]:
                for label, value in zip(['groups','mixed_fraction','accuracy','all_wrong_fraction',
                                         'all_correct_fraction','truncation_rate'], values):
                    metrics[f'curriculum/{key}/{label}'] = value if label=='groups' else value/values[0]
        metrics.update({f'curriculum/pool/{k}/questions':v for k,v in self.last_pool_counts.items()})
        self.completed_step, self.pending_step = step, None
        if step-self.origin_step == self.spec['refresh_after_updates']:
            self.frozen = {}
            prior_count = self.spec['prior_groups']
            for key, values in self.stats.items():
                prior = self.spec['strata'][key]
                mixed = (values[1] + prior_count*prior['mixed_fraction'])/(values[0]+prior_count)
                trunc = (values[5] + prior_count*prior['truncation_rate'])/(values[0]+prior_count)
                self.frozen[key] = max(0.1, mixed*(1-trunc))
                metrics[f'curriculum/{key}/next_phase_weight'] = self.frozen[key]
        return metrics

    def state_dict(self):
        if self.pending_step is not None:
            raise ValueError('Cannot checkpoint before observations are complete')
        return dict(origin_step=self.origin_step, completed_step=self.completed_step,
                    positions=self.positions, stats=self.stats, frozen=self.frozen)
