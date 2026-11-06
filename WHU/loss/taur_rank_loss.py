"""Camera-conditioned, all-negative-image ranking in the unified pre-BN space."""
import torch
from torch import nn
from torch.nn import functional as F


class TAURRankingLoss(nn.Module):
    DIRECTIONS = ('T2T', 'T2ALL', 'R2T', 'N2T')

    def __init__(self, margin=0.03, temperature=0.07):
        super().__init__()
        if temperature <= 0 or margin < 0:
            raise ValueError('TAUR temperature must be positive and margin nonnegative')
        self.margin, self.temperature = margin, temperature

    def forward(self, features, pid, camid, modality_id, sample_id):
        n = features.shape[0]
        if features.ndim != 2 or any(t.ndim != 1 or t.numel() != n
                                   for t in (pid, camid, modality_id, sample_id)):
            raise ValueError('TAUR requires one metadata row per feature')
        # Explicitly disable autocast: float() alone does not protect a matmul.
        with torch.autocast(device_type=features.device.type, enabled=False):
            f = F.normalize(features.float(), dim=1)
            similarity = f @ f.t()
            eligible = (camid[:, None] != camid[None, :]) & (
                sample_id[:, None] != sample_id[None, :])
            same_pid = pid[:, None] == pid[None, :]
            zero = features.float().sum() * 0.0
            losses, stats = {}, {}
            for direction, query_mod, gallery_mod in (
                    ('T2T', 3, 3), ('T2ALL', 3, None),
                    ('R2T', 1, 3), ('N2T', 2, 3)):
                gallery = torch.ones_like(modality_id, dtype=torch.bool) if gallery_mod is None else modality_id == gallery_mod
                candidates = eligible & gallery[None, :]
                pos = candidates & same_pid
                neg = candidates & ~same_pid
                valid = (modality_id == query_mod) & pos.any(1) & neg.any(1)
                rows = valid.nonzero(as_tuple=True)[0]
                if rows.numel():
                    s, p, q = similarity[rows], pos[rows], neg[rows]
                    # log(1 + sum exp((s_neg - s_pos + margin) / tau))
                    neg_lse = torch.logsumexp((s / self.temperature).masked_fill(~q, -torch.inf), dim=1)
                    pair = self.temperature * F.softplus(
                        neg_lse[:, None] + (self.margin - s) / self.temperature)
                    query_loss = pair.masked_fill(~p, 0).sum(1) / p.sum(1)
                    losses[direction] = query_loss.mean()
                else:
                    losses[direction] = zero
                pos_counts, neg_counts = pos[valid].sum(1), neg[valid].sum(1)
                positive_cameras = sum((pos[valid] & (camid == camera)[None, :]).any(1).sum()
                                       for camera in camid.unique())
                stats[direction] = {
                    'queries': int(rows.numel()),
                    'possible_queries': int((modality_id == query_mod).sum()),
                    'positive_pairs': int(pos_counts.sum()),
                    'negative_pairs': int(neg_counts.sum()),
                    'positive_camera_sum': int(positive_cameras),
                }
            # Missing directions keep their fixed 0.25 coefficients.
            return sum(losses.values()) * 0.25, losses, stats


def rank_weight(epoch, cfg):
    if epoch <= cfg.WARMUP_EPOCHS:
        return 0.0
    return cfg.RANK_WEIGHT * min(1.0, (epoch - cfg.WARMUP_EPOCHS) /
                                (cfg.RAMP_END_EPOCH - cfg.WARMUP_EPOCHS))
