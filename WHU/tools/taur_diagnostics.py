"""Read-only checkpoint diagnostics. No training imports or optimizer operations."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

STATES = ('uad', 'taur_on', 'taur_off')
META = ('pid', 'camid', 'modid', 'platform', 'path', 'split', 'index', 'modality_index')
NAMES = {1: 'RGB', 2: 'NIR', 3: 'TIR'}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata(loaders, query_counts):
    """Recover full paths from original dataset rows; never infer modid from mode."""
    if len(loaders) != len(query_counts):
        raise ValueError('Loader/query-count length mismatch')
    grouped = {'query': [], 'gallery': []}
    for loader, nq in zip(loaders, query_counts):
        ds = loader.dataset
        rows = ds.dataset[ds.modality]
        if not 0 <= nq <= len(rows):
            raise ValueError('Invalid query boundary')
        for i, (path, pid, camid, mid) in enumerate(rows):
            if camid not in range(7) or mid not in NAMES:
                raise ValueError('Unknown WHU camera/modality ID')
            split = 'query' if i < nq else 'gallery'
            grouped[split].append((pid, camid, mid, 'G' if camid < 5 else 'A',
                                   str(path), split, len(grouped[split]),
                                   i if split == 'query' else i - nq))
    rows = grouped['query'] + grouped['gallery']
    if not grouped['query'] or not grouped['gallery']:
        raise ValueError('Empty query or gallery')
    return {key: np.asarray([r[j] for r in rows]) for j, key in enumerate(META)}


def export_features(model, loaders, query_counts, device):
    """Original validation preprocessing/forward, FP32, eval and inference_mode."""
    meta = metadata(loaders, query_counts)
    groups = {'query': [], 'gallery': []}
    model.eval()
    before = state_hash(model)
    with torch.inference_mode():
        for mode, (loader, nq) in enumerate(zip(loaders, query_counts), 1):
            ds = loader.dataset
            rows = ds.dataset[ds.modality]
            offset, chunks = 0, []
            for images, pids, cams, _, mids, filenames in loader:
                expected = rows[offset:offset + len(images)]
                actual = list(zip(map(int, pids), map(int, cams), map(int, mids), filenames))
                wanted = [(int(p), int(c), int(m), Path(path).name) for path, p, c, m in expected]
                if actual != wanted:
                    raise ValueError('Validation loader order/metadata mismatch')
                feature = model(images.to(device), mode=mode,
                                modality_ids=torch.as_tensor(mids, device=device))
                if feature.ndim != 2 or feature.shape != (len(images), 768):
                    raise ValueError('Expected 768-dimensional pre-BN CLS')
                chunks.append(feature.float().cpu())
                offset += len(images)
            if offset != len(rows):
                raise ValueError('Incomplete validation loader')
            features = torch.cat(chunks) if chunks else torch.empty(0, 768)
            groups['query'].append(features[:nq])
            groups['gallery'].append(features[nq:])
    if state_hash(model) != before:
        raise AssertionError('Inference modified checkpoint parameters/buffers')
    meta['features'] = torch.cat(groups['query'] + groups['gallery']).numpy()
    return meta


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        pack = {key: data[key] for key in ('features',) + META}
    n = len(pack['features'])
    if pack['features'].shape != (n, 768) or not np.isfinite(pack['features']).all():
        raise ValueError('Invalid feature matrix')
    if np.any(np.linalg.norm(pack['features'], axis=1) == 0):
        raise ValueError('Zero feature vector')
    if any(pack[key].shape != (n,) for key in META):
        raise ValueError('Invalid metadata length')
    if not np.isin(pack['split'], ['query', 'gallery']).all():
        raise ValueError('Invalid split')
    if not np.isin(pack['modid'], [1, 2, 3]).all() or not np.isin(pack['camid'], range(7)).all():
        raise ValueError('Invalid WHU modality/camera')
    if not np.array_equal(pack['platform'], np.where(pack['camid'] < 5, 'G', 'A')):
        raise ValueError('Platform disagrees with actual camera IDs')
    for split in ('query', 'gallery'):
        idx = pack['split'] == split
        if not idx.any() or not np.array_equal(pack['index'][idx], np.arange(idx.sum())):
            raise ValueError('Invalid original split order')
        if len(np.unique(pack['path'][idx])) != idx.sum():
            raise ValueError('Duplicate image paths in split')
    return pack


def assert_aligned(packs):
    reference = packs['uad']
    for state in STATES[1:]:
        for key in META:
            if not np.array_equal(reference[key], packs[state][key]):
                raise ValueError('Unpaired {} metadata: {}'.format(state, key))


def feature_effect(packs, atol=1e-6, rtol=1e-5):
    on, off = packs['taur_on'], packs['taur_off']
    rows = []
    for mid in NAMES:
        mask = on['modid'] == mid
        if mid != 3 and not np.allclose(on['features'][mask], off['features'][mask], atol=atol, rtol=rtol):
            raise AssertionError('RGB/NIR changed when bypassing adapters; stop analysis')
        for split in ('query', 'gallery'):
            idx = mask & (on['split'] == split)
            if not idx.any():
                continue
            a = F.normalize(torch.from_numpy(on['features'][idx]).double(), dim=1)
            b = F.normalize(torch.from_numpy(off['features'][idx]).double(), dim=1)
            delta = (1 - (a * b).sum(1)).clamp_min(0).numpy()
            rows.append(dict(modality=NAMES[mid], split=split, count=int(idx.sum()),
                             delta_mean=float(delta.mean()), delta_median=float(np.median(delta)),
                             delta_p90=float(np.quantile(delta, .9)),
                             max_abs_raw_difference=float(np.abs(on['features'][idx] - off['features'][idx]).max()),
                             atol=atol, rtol=rtol))
    return rows


def protocols(pack):
    q, g = pack['split'] == 'query', pack['split'] == 'gallery'
    result = [('ALL->ALL', q, g)]
    result += [(NAMES[a] + '->' + NAMES[b], q & (pack['modid'] == a), g & (pack['modid'] == b))
               for a in NAMES for b in NAMES]
    result.append(('TIR->ALL', q & (pack['modid'] == 3), g))
    for a in ('G', 'A'):
        for b in ('G', 'A'):
            result.append(('TIR-' + a + '->TIR-' + b,
                           q & (pack['modid'] == 3) & (pack['platform'] == a),
                           g & (pack['modid'] == 3) & (pack['platform'] == b)))
    return result


def query_statistics(pack, qi, ordered, similarities):
    """Complete ranking after exclusion of ALL same-camera gallery images.

    V2 counts follow ordered ranks. V3 C_img/C_id use strict score >, so
    tied negatives do not count as outranking positives in those two fields.
    """
    keep = pack['camid'][ordered] != pack['camid'][qi]
    ordered, scores = ordered[keep], similarities[ordered[keep]]
    positive = pack['pid'][ordered] == pack['pid'][qi]
    positions = np.flatnonzero(positive)
    if not len(positions):
        return None
    ranks = positions + 1
    npos = len(positions)
    negatives = np.flatnonzero(~positive)
    neg_scores = scores[~positive]
    pos_scores = scores[positive]
    neg_pids = pack['pid'][ordered[~positive]]
    _, first_negative = np.unique(neg_pids, return_index=True)
    first_ranks = np.sort(negatives[first_negative])
    wrong_ids = np.searchsorted(first_ranks, positions, side='left')
    cimg = len(neg_scores) - np.searchsorted(np.sort(neg_scores), pos_scores, side='right')
    # Reduce each negative identity to its maximum similarity for strict C_id.
    unique_ids, inverse = np.unique(neg_pids, return_inverse=True)
    maxima = np.full(len(unique_ids), -np.inf)
    np.maximum.at(maxima, inverse, neg_scores)
    cid = len(maxima) - np.searchsorted(np.sort(maxima), pos_scores, side='right')
    best_pos = ordered[positions[np.argmax(pos_scores)]]
    best_neg = ordered[negatives[np.argmax(neg_scores)]] if len(negatives) else None
    maxneg = float(neg_scores.max()) if len(negatives) else float('nan')
    row = dict(query_index=int(pack['index'][qi]), query_path=str(pack['path'][qi]),
               pid=int(pack['pid'][qi]), query_camid=int(pack['camid'][qi]),
               query_modality=NAMES[int(pack['modid'][qi])], query_platform=str(pack['platform'][qi]),
               first_pos_rank=int(ranks[0]), AP=float((np.arange(1, npos + 1) / ranks).mean()),
               INP=float(npos / ranks[-1]), last_pos_rank=int(ranks[-1]), num_valid_pos=npos,
               neg_before_pos_mean=float((ranks - np.arange(1, npos + 1)).mean()),
               neg_before_last_pos=int(ranks[-1] - npos), wrong_id_before_pos_mean=float(wrong_ids.mean()),
               C_img=float(cimg.mean()), C_id=float(cid.mean()),
               M_best=float(pos_scores.max() - maxneg), M_med=float(np.median(pos_scores) - maxneg),
               best_positive_modality=NAMES[int(pack['modid'][best_pos])],
               best_positive_platform=str(pack['platform'][best_pos]), best_positive_camid=int(pack['camid'][best_pos]),
               best_positive_path=str(pack['path'][best_pos]),
               most_similar_negative_modality=NAMES[int(pack['modid'][best_neg])] if best_neg is not None else '',
               most_similar_negative_platform=str(pack['platform'][best_neg]) if best_neg is not None else '',
               most_similar_negative_camid=int(pack['camid'][best_neg]) if best_neg is not None else '',
               most_similar_negative_path=str(pack['path'][best_neg]) if best_neg is not None else '',
               top1_wrong=bool(not positive[0]), top1_modality=NAMES[int(pack['modid'][ordered[0]])],
               score_ties=int(len(scores) - len(np.unique(scores))))
    for mid, name in NAMES.items():
        mask = pack['modid'][ordered] == mid
        row['valid_pos_' + name] = int((mask & positive).sum())
        row['valid_neg_' + name] = int((mask & ~positive).sum())
    return row


def evaluate_protocol(pack, qmask, gmask, chunk_size=32):
    """FP32 normalized squared Euclidean distance and torch.argsort, as UAD."""
    qidx, gidx = np.flatnonzero(qmask), np.flatnonzero(gmask)
    if not len(qidx) or not len(gidx):
        return []
    features = F.normalize(torch.from_numpy(pack['features']).float(), dim=1)
    gf = features[gidx]
    gsq = gf.square().sum(1)[None, :]
    output = []
    for start in range(0, len(qidx), chunk_size):
        qi = qidx[start:start + chunk_size]
        qf = features[qi]
        distance = qf.square().sum(1)[:, None] + gsq
        distance.addmm_(qf, gf.t(), beta=1, alpha=-2)
        orders = torch.argsort(distance, dim=1).numpy()
        sim = (qf @ gf.t()).numpy()
        for j, index in enumerate(qi):
            # Indexed full vector keeps statistics independent of gallery subset size.
            scores = np.zeros(len(features), dtype=np.float32)
            scores[gidx] = sim[j]
            row = query_statistics(pack, index, gidx[orders[j]], scores)
            if row is not None:
                output.append(row)
    return output


def aggregate(rows):
    if not rows:
        return dict(valid_queries=0, mAP=float('nan'), mINP=float('nan'),
                    rank1=float('nan'), rank5=float('nan'), rank10=float('nan'))
    result = dict(valid_queries=len(rows), mAP=100 * np.mean([r['AP'] for r in rows]),
                  mINP=100 * np.mean([r['INP'] for r in rows]))
    for k in (1, 5, 10):
        result['rank' + str(k)] = 100 * np.mean([r['first_pos_rank'] <= k for r in rows])
    return result
