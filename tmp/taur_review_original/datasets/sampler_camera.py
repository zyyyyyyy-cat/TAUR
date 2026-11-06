"""Camera-priority PKM, optional for TAUR and its matched UAD control."""
from collections import defaultdict
import random
from torch.utils.data import Sampler


class CameraPKMSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances, modalities,
                 seed=1234, rank=0, world_size=1):
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

    def _sample_group(self, pid, modality, rng):
        by_camera = defaultdict(list)
        pool = self.indices[pid][modality]
        for index in pool:
            by_camera[self.data_source[modality][index][2]].append(index)
        cameras = list(by_camera)
        rng.shuffle(cameras)
        chosen = [rng.choice(by_camera[c]) for c in cameras[:self.k]]
        remaining = [i for i in pool if i not in chosen]
        chosen.extend(rng.sample(remaining, min(self.k - len(chosen), len(remaining))))
        while len(chosen) < self.k:
            chosen.append(rng.choice(pool))
        rng.shuffle(chosen)
        return chosen

    def __iter__(self):
        # Private RNG: all ranks build the same global plan, independently of workers.
        rng = random.Random(self.seed + self.epoch)
        remaining = {pid: min(count, self.steps) for pid, count in self.groups.items()}
        local = []
        for _ in range(self.steps):
            candidates = list(self.pids)
            rng.shuffle(candidates)
            candidates.sort(key=lambda pid: remaining[pid], reverse=True)
            selected = candidates[:self.p]
            if any(remaining[pid] <= 0 for pid in selected):
                raise RuntimeError('Invalid complete PKM batch plan')
            rng.shuffle(selected)
            global_batch = []
            for pid in selected:
                remaining[pid] -= 1
                groups = [self._sample_group(pid, m, rng) for m in self.modalities]
                global_batch.extend(zip(*groups))
            begin = self.rank * self.local_batch
            local.extend(global_batch[begin:begin + self.local_batch])
        return iter(local)

    def __len__(self):
        return self.steps * self.local_batch
