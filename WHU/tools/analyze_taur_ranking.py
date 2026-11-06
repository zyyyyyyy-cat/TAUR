"""V1/V2/V3 analysis using existing complete features; never run a model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taur_diagnostics import (STATES, META, load_features, assert_aligned,
                              feature_effect, protocols, evaluate_protocol,
                              aggregate, write_csv, file_hash)

PAIRS = (('uad', 'taur_on'), ('taur_off', 'taur_on'), ('uad', 'taur_off'))
FIELDS = ('first_pos_rank', 'AP', 'INP', 'last_pos_rank', 'num_valid_pos',
          'neg_before_pos_mean', 'neg_before_last_pos', 'wrong_id_before_pos_mean',
          'C_img', 'C_id', 'M_best', 'M_med')


def transition(a, b):
    return ('correct' if a['first_pos_rank'] == 1 else 'wrong') + '->' + (
        'correct' if b['first_pos_rank'] == 1 else 'wrong')


def trend(value, higher=True):
    if abs(value) < 1e-12:
        return 'unchanged'
    return 'improved' if (value > 0) == higher else 'worsened'


def paired_bootstrap(a, b, samples, seed):
    """Resample query PID clusters, retaining query-weighted metric means."""
    if not a or samples <= 0:
        return {}
    pids = np.asarray([r['pid'] for r in a])
    unique, inv = np.unique(pids, return_inverse=True)
    counts = np.bincount(inv)
    delta = np.asarray([[y['AP'] - x['AP'],
                         float(y['first_pos_rank'] == 1) - float(x['first_pos_rank'] == 1)]
                        for x, y in zip(a, b)])
    sums = np.stack([np.bincount(inv, weights=delta[:, j]) for j in range(2)], 1)
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(samples):
        selected = rng.randint(0, len(unique), len(unique))
        result.append(100 * sums[selected].sum(0) / counts[selected].sum())
    intervals = np.quantile(result, [.025, .975], axis=0)
    return dict(bootstrap_unit='query_pid', bootstrap_samples=samples, query_identities=len(unique),
                delta_mAP_ci_low=intervals[0, 0], delta_mAP_ci_high=intervals[1, 0],
                delta_rank1_ci_low=intervals[0, 1], delta_rank1_ci_high=intervals[1, 1])


def compare_rows(results, samples, seed):
    transitions, summary = [], []
    for protocol in results['uad']:
        states = {s: results[s][protocol] for s in STATES}
        ids = [r['query_index'] for r in states['uad']]
        if any([r['query_index'] for r in states[s]] != ids for s in STATES):
            raise AssertionError('Valid query sets differ across states')
        for i, reference in enumerate(states['uad']):
            row = {k: reference[k] for k in ('query_index', 'query_path', 'pid', 'query_camid',
                                             'query_modality', 'query_platform')}
            row['protocol'] = protocol
            for state in STATES:
                for key in FIELDS:
                    row[state + '_' + key] = states[state][i][key]
                for key in ('top1_modality', 'best_positive_modality', 'best_positive_platform'):
                    row[state + '_' + key] = states[state][i][key]
            for old, new in PAIRS:
                a, b = states[old][i], states[new][i]
                label = new + '_minus_' + old
                row[label + '_rank1_transition'] = transition(a, b)
                for key in FIELDS:
                    row[label + '_' + key] = b[key] - a[key]
            transitions.append(row)
        for old, new in PAIRS:
            for group in ('ALL', 'RGB', 'NIR', 'TIR', 'G', 'A', 'TIR_best_positive_gallery',
                          'cross_platform_best_positive'):
                indices = [i for i, r in enumerate(states[old]) if group == 'ALL' or
                           r['query_modality'] == group or r['query_platform'] == group or
                           (group == 'TIR_best_positive_gallery' and
                            (r['best_positive_modality'] == 'TIR' or states[new][i]['best_positive_modality'] == 'TIR')) or
                           (group == 'cross_platform_best_positive' and
                            (r['best_positive_platform'] != r['query_platform'] or
                             states[new][i]['best_positive_platform'] != r['query_platform']))]
                a, b = [states[old][i] for i in indices], [states[new][i] for i in indices]
                row = dict(protocol=protocol, comparison=new + '_minus_' + old, group=group, queries=len(a))
                for category in ('correct->correct', 'correct->wrong', 'wrong->correct', 'wrong->wrong'):
                    selected = [(x, y) for x, y in zip(a, b) if transition(x, y) == category]
                    row[category + '_count'] = len(selected)
                    row[category + '_mean_delta_AP'] = np.mean([y['AP'] - x['AP'] for x, y in selected]) if selected else float('nan')
                    row[category + '_AP_improved_count'] = sum(y['AP'] > x['AP'] + 1e-12 for x, y in selected)
                row['net_rank1_repairs'] = row['wrong->correct_count'] - row['correct->wrong_count']
                for rt in ('improved', 'worsened', 'unchanged'):
                    for apt in ('improved', 'worsened', 'unchanged'):
                        row['first_' + rt + '_AP_' + apt] = sum(
                            trend(y['first_pos_rank'] - x['first_pos_rank'], False) == rt and
                            trend(y['AP'] - x['AP']) == apt for x, y in zip(a, b))
                for key in FIELDS:
                    row['mean_delta_' + key] = np.mean([y[key] - x[key] for x, y in zip(a, b)]) if a else float('nan')
                row['first_earlier_last_later_count'] = sum(y['first_pos_rank'] < x['first_pos_rank'] and
                                                          y['last_pos_rank'] > x['last_pos_rank'] for x, y in zip(a, b))
                row['AP_improved_count'] = sum(y['AP'] > x['AP'] + 1e-12 for x, y in zip(a, b))
                row['AP_worsened_count'] = sum(y['AP'] < x['AP'] - 1e-12 for x, y in zip(a, b))
                row['last_pos_later_count'] = sum(y['last_pos_rank'] > x['last_pos_rank'] for x, y in zip(a, b))
                row['neg_before_last_increased_count'] = sum(y['neg_before_last_pos'] > x['neg_before_last_pos'] for x, y in zip(a, b))
                row['best_margin_up_median_down_count'] = sum(y['M_best'] > x['M_best'] and y['M_med'] < x['M_med'] for x, y in zip(a, b))
                row['both_margins_up_AP_up_count'] = sum(y['M_best'] > x['M_best'] and y['M_med'] > x['M_med'] and y['AP'] > x['AP'] for x, y in zip(a, b))
                row['both_margins_up_AP_down_count'] = sum(y['M_best'] > x['M_best'] and y['M_med'] > x['M_med'] and y['AP'] < x['AP'] for x, y in zip(a, b))
                row['both_margins_down_count'] = sum(y['M_best'] < x['M_best'] and y['M_med'] < x['M_med'] for x, y in zip(a, b))
                row['both_error_counts_down_AP_down_count'] = sum(y['C_img'] < x['C_img'] and y['C_id'] < x['C_id'] and y['AP'] < x['AP'] for x, y in zip(a, b))
                row['delta_rank1_pp'] = 100 * row['net_rank1_repairs'] / len(a) if a else float('nan')
                row['delta_mAP_pp'] = 100 * row['mean_delta_AP']
                row.update(paired_bootstrap(a, b, samples, seed))
                summary.append(row)
    return transitions, summary


def run_analysis(directory, chunk_size=32, bootstrap=1000, seed=1234, plot=True):
    directory = Path(directory)
    if chunk_size <= 0 or bootstrap < 0:
        raise ValueError('chunk-size must be positive and bootstrap nonnegative')
    manifest_path = directory / 'diagnostic_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('evaluation') != {
            'TOP_K_EVAL': 0, 'RE_RANKING': False, 'NECK_FEAT': 'before', 'FEAT_NORM': True,
            'same_camera_filter': 'all_gallery', 'precision': 'float32'}:
        raise ValueError('Unknown/incompatible extraction manifest')
    if manifest['states']['taur_on']['checkpoint_sha256'] != manifest['states']['taur_off']['checkpoint_sha256']:
        raise ValueError('TAUR on/off must use the same checkpoint')
    packs = {}
    for state in STATES:
        path = directory / (state + '_features.npz')
        if file_hash(path) != manifest['states'][state]['feature_sha256']:
            raise ValueError('Feature hash mismatch: ' + state)
        packs[state] = load_features(path)
    assert_aligned(packs)
    effect = feature_effect(packs)  # stop before producing comparisons if RGB/NIR differs
    results, comparisons, platform, mechanism, mixed = {}, [], [], [], []
    for state in STATES:
        results[state] = {}
        for name, qmask, gmask in protocols(packs[state]):
            print('Evaluating {} {} (complete gallery)'.format(state, name), flush=True)
            rows = evaluate_protocol(packs[state], qmask, gmask, chunk_size)
            results[state][name] = rows
            metrics = dict(state=state, protocol=name, query_count=int(qmask.sum()),
                           gallery_count=int(gmask.sum()), **aggregate(rows))
            (platform if name.startswith(('TIR-G', 'TIR-A')) else comparisons).append(metrics)
            if name in ('TIR->TIR', 'TIR->ALL', 'RGB->TIR', 'NIR->TIR', 'RGB->RGB', 'TIR->RGB', 'TIR->NIR'):
                mechanism.extend(dict(state=state, protocol=name, **r) for r in rows)
            if name == 'TIR->ALL':
                wrong = [r for r in rows if r['top1_wrong']]
                info = dict(state=state, protocol=name, **aggregate(rows), wrong_top1_queries=len(wrong),
                            wrong_top1_fraction=len(wrong) / len(rows) if rows else float('nan'),
                            mean_C_img=np.mean([r['C_img'] for r in rows]) if rows else float('nan'))
                for mod in ('RGB', 'NIR', 'TIR'):
                    info['wrong_top1_' + mod + '_fraction'] = sum(r['top1_modality'] == mod for r in wrong) / len(wrong) if wrong else float('nan')
                mixed.append(info)
    transitions, summary = compare_rows(results, bootstrap, seed)
    write_csv(directory / 'checkpoint_comparison.csv', comparisons)
    write_csv(directory / 'adapter_feature_effect.csv', effect)
    write_csv(directory / 'query_rank_transitions.csv', transitions)
    write_csv(directory / 'ranking_summary.csv', summary)
    write_csv(directory / 'tir_ranking_mechanism.csv', mechanism)
    write_csv(directory / 'tir_platform_breakdown.csv', platform)
    write_csv(directory / 'tir_mixed_gallery_summary.csv', mixed)
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib unavailable; CSV analysis completed, plot skipped', flush=True)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            for category, color in [('TIR query', '#c23b22'), ('RGB/NIR query, TIR best positive', '#e69f00'),
                                    ('RGB/NIR query, RGB/NIR best positive', '#2878b5')]:
                selected = []
                for r in transitions:
                    if r['protocol'] != 'ALL->ALL':
                        continue
                    actual = 'TIR query' if r['query_modality'] == 'TIR' else (
                        'RGB/NIR query, TIR best positive' if r['uad_best_positive_modality'] == 'TIR' or
                        r['taur_on_best_positive_modality'] == 'TIR' else 'RGB/NIR query, RGB/NIR best positive')
                    if actual == category:
                        selected.append(r)
                ax.scatter([r['taur_on_minus_uad_first_pos_rank'] for r in selected],
                           [100 * r['taur_on_minus_uad_AP'] for r in selected],
                           s=9, alpha=.45, color=color, label=category)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.axvline(0, color='gray', linewidth=.7)
            ax.set_xlabel('First positive rank: TAUR on - UAD (lower is better)')
            ax.set_ylabel('AP change (percentage points)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory / 'rank_ap_changes.png', dpi=180)
            plt.close(fig)
    manifest['analysis'] = dict(chunk_size=chunk_size, bootstrap_samples=bootstrap, bootstrap_seed=seed,
                                bootstrap_unit='query_pid', full_gallery=True,
                                sorting='FP32 normalized squared Euclidean; torch.argsort',
                                tie_policy='torch default ordering; V3 C_img/C_id use strict similarity >',
                                limitation='Fixed checkpoints/query sampling only; no across-training-seed inference')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Diagnostics completed: ' + str(directory), flush=True)
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='diagnostics/TAUR_vs_UAD')
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    run_analysis(args.output_dir, args.chunk_size, args.bootstrap, args.seed, not args.no_plot)
