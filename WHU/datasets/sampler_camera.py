"""Camera-priority PKM, optional for TAUR and its matched UAD control."""
from collections import defaultdict
import random
from torch.utils.data import Sampler


class CameraPKMSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, modalities,
                 seed=1234, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid PKM rank/world_size')
        if batch_size <= 0 or num_instances <= 0 or batch_size % (world_size * num_instances):
            raise ValueError('Global PKM tuple batch must be divisible by WORLD_SIZE * K')
        self.data_source, self.modalities = data_source, list(modalities)
        self.batch_size, self.k = batch_size, num_instances
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0
        self.local_batch = batch_size // world_size
        self.p = batch_size // num_instances
        self.indices = defaultdict(lambda: defaultdict(list))
        for m in self.modalities:
            for i, (_, pid, _, _) in enumerate(data_source[m]):
                self.indices[pid][m].append(i)
        self.pids = sorted(self.indices)
        self.groups = {}
        for pid in self.pids:
            if any(not self.indices[pid][m] for m in self.modalities):
                raise ValueError('PKM identity {} is missing a modality'.format(pid))
            self.groups[pid] = max(1, max(len(self.indices[pid][m])
                                         for m in self.modalities) // self.k)
        # Maximum complete batches with distinct identities, with no padding.
        lo, hi = 0, sum(self.groups.values()) // self.p
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sum(min(g, mid) for g in self.groups.values()) >= self.p * mid:
                lo = mid
            else:
                hi = mid - 1
        self.steps = lo
        if self.steps == 0:
            raise ValueError('Not enough distinct identities for a complete global PKM batch')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _sample_groups(self, pid, modality, count, rng):
        # Build one epoch pool, as in UAD; repeat only undersized modalities.
        need = count * self.k
        source = self.indices[pid][modality]
        copies, tail = divmod(need, len(source))
        pool = list(source) * copies + rng.sample(source, tail)
        by_camera = defaultdict(list)
        for index in pool:
            by_camera[self.data_source[modality][index][2]].append(index)
        for queue in by_camera.values():
            rng.shuffle(queue)
        groups = []
        for _ in range(count):
            cameras = [c for c, queue in by_camera.items() if queue]
            rng.shuffle(cameras)
            chosen = [by_camera[c].pop() for c in cameras[:self.k]]
            while len(chosen) < self.k:
                candidates = [(c, pos) for c, queue in by_camera.items()
                              for pos, index in enumerate(queue) if index not in chosen]
                if not candidates:
                    candidates = [(c, pos) for c, queue in by_camera.items()
                                  for pos in range(len(queue))]
                camera, position = rng.choice(candidates)
                chosen.append(by_camera[camera].pop(position))
            rng.shuffle(chosen)
            groups.append(chosen)
        # Scarce cameras may exhaust: do not recycle images merely to fake coverage.
        rng.shuffle(groups)
        return groups

    def __iter__(self):
        # Private RNG: all ranks build the same global plan, independently of workers.
        rng = random.Random(self.seed + self.epoch)
        remaining = {pid: min(count, self.steps) for pid, count in self.groups.items()}
        surplus = sum(remaining.values()) - self.steps * self.p
        for _ in range(surplus):
            candidates = [pid for pid, count in remaining.items() if count > 1]
            if not candidates:
                candidates = [pid for pid, count in remaining.items() if count]
            remaining[rng.choice(candidates)] -= 1
        quotas = dict(remaining)
        batches = []
        for steps_left in range(self.steps, 0, -1):
            # Only deadline-constrained identities are mandatory. Others follow
            # random PKM selection, rather than a largest-count curriculum.
            mandatory = [pid for pid, count in remaining.items() if count == steps_left]
            candidates = [pid for pid, count in remaining.items()
                          if 0 < count < steps_left]
            selected = mandatory + rng.sample(candidates, self.p - len(mandatory))
            for pid in selected:
                remaining[pid] -= 1
            batches.append(selected)
        rng.shuffle(batches)
        pools = {pid: {m: self._sample_groups(pid, m, count, rng) for m in self.modalities}
                 for pid, count in quotas.items() if count}
        local = []
        unique = {m: set() for m in self.modalities}
        for selected in batches:
            rng.shuffle(selected)
            global_batch = []
            for pid in selected:
                groups = [pools[pid][m].pop() for m in self.modalities]
                for m, group in zip(self.modalities, groups):
                    unique[m].update(group)
                global_batch.extend(zip(*groups))
            begin = self.rank * self.local_batch
            local.extend(global_batch[begin:begin + self.local_batch])
        self.last_stats = {'global_steps': self.steps,
                           'draws_per_modality': self.steps * self.batch_size,
                           'unique_images': {m: len(indices) for m, indices in unique.items()},
                           'identities': sum(count > 0 for count in quotas.values())}
        return iter(local)

    def __len__(self):
        return self.steps * self.local_batch
