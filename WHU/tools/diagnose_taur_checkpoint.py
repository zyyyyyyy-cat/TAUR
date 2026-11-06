"""Extract UAD, TAUR-on and TAUR-off once, then run offline V1/V2/V3."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from config import cfg as defaults
from datasets import make_dataloader
from model import make_model
from test import setup_cuda_visible_devices
from taur_diagnostics import (STATES, META, metadata, export_features,
                              file_hash, load_features)
from analyze_taur_ranking import run_analysis

EVALUATION = dict(TOP_K_EVAL=0, RE_RANKING=False, NECK_FEAT='before', FEAT_NORM=True,
                  same_camera_filter='all_gallery', precision='float32')


def extraction_config(path, opts, taur):
    cfg = defaults.clone()
    cfg.defrost()
    cfg.merge_from_file(path)
    cfg.merge_from_list(opts)
    # Diagnostic-only cloned configuration: never change normal train/test config.
    cfg.MODEL.DIST_TRAIN = False
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.TAUR.ENABLED = taur
    cfg.TEST.TOP_K_EVAL = 0
    cfg.TEST.RE_RANKING = False
    cfg.TEST.NECK_FEAT = 'before'
    cfg.TEST.FEAT_NORM = 'yes'
    if list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']:
        raise ValueError('Diagnostics require the original RGB/IR/Thermal order')
    cfg.freeze()
    return cfg


def extract(args):
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('Use one process, not torchrun: these are inference diagnostics')
    configs = {s: extraction_config(args.uad_config if s == 'uad' else args.taur_config,
                                    args.opts, s != 'uad') for s in STATES}
    u, t = configs['uad'], configs['taur_on']
    if u.INPUT.SIZE_TEST != t.INPUT.SIZE_TEST or u.INPUT.PIXEL_MEAN != t.INPUT.PIXEL_MEAN or u.INPUT.PIXEL_STD != t.INPUT.PIXEL_STD or u.DATASETS != t.DATASETS:
        raise ValueError('UAD/TAUR evaluation preprocessing or dataset settings differ')
    setup_cuda_visible_devices(t)
    device = torch.device(t.MODEL.DEVICE)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use MODEL.DEVICE cpu for CPU extraction')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'diagnostic_manifest.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if args.reuse and manifest_path.exists() else {}
    if previous and (previous.get('schema_version') != 1 or previous.get('evaluation') != EVALUATION):
        raise ValueError('Incompatible existing manifest')
    manifest = dict(schema_version=1, evaluation=EVALUATION, feature_dimension=768,
                    platform_rule='WHU zero-based cameras 0-4 Ground; 5-6 Aerial',
                    versions=dict(torch=torch.__version__, numpy=np.__version__),
                    states=dict(previous.get('states', {})))
    # Use the original validation loaders, including exact resize/normalize and split ordering.
    # make_dataloader constructs train loaders but diagnostics never iterate them.
    _, _, loaders, counts, classes, cameras, views = make_dataloader(t)
    expected = metadata(loaders, counts)
    model = None
    loaded_state = None
    for state in STATES:
        checkpoint = Path(args.uad_weight if state == 'uad' else args.taur_weight).resolve()
        config = configs[state]
        signature = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                         adapter_enabled=state != 'taur_off', taur_architecture=state != 'uad',
                         config=config.dump())
        output = out / (state + '_features.npz')
        cached = previous.get('states', {}).get(state, {})
        if args.reuse and output.exists():
            if any(cached.get(k) != v for k, v in signature.items()) or cached.get('feature_sha256') != file_hash(output):
                raise ValueError('Cannot reuse unverified/stale cache: ' + str(output))
            pack = load_features(output)
            if any(not np.array_equal(pack[k], expected[k]) for k in META):
                raise ValueError('Cached dataset order/metadata differs: ' + state)
            print('Reusing verified ' + str(output), flush=True)
        else:
            if state != 'taur_off' or loaded_state != 'taur_on':
                if model is not None:
                    del model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                model = make_model(config, num_class=classes, camera_num=cameras, view_num=views)
                model.load_param(str(checkpoint))
                model.to(device).eval()
                loaded_state = state
            # When on is freshly extracted, off uses that exact loaded model.
            # When on was cached, load the same TAUR checkpoint for off.
            if state == 'taur_off' and not model.base.thermal_adapters:
                raise AssertionError('TAUR-off requires a TAUR checkpoint/model')
            model.base.adapter_enabled = state != 'taur_off'
            print('Extracting {}: {}'.format(state, checkpoint), flush=True)
            pack = export_features(model, loaders, counts, device)
            temporary = output.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, **pack)
            temporary.replace(output)
        signature.update(feature_file=output.name, feature_sha256=file_hash(output),
                         query_count=int((pack['split'] == 'query').sum()),
                         gallery_count=int((pack['split'] == 'gallery').sum()))
        manifest['states'][state] = signature
        # Checkpoint-level manifest permits resuming interrupted feature extraction.
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if model is not None:
        model.base.adapter_enabled = True
    if not args.extract_only:
        run_analysis(out, args.chunk_size, args.bootstrap, args.seed, not args.no_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uad-weight', required=True)
    parser.add_argument('--taur-weight', required=True)
    parser.add_argument('--uad-config', default='configs/UAD.yml')
    parser.add_argument('--taur-config', default='configs/TAUR.yml')
    parser.add_argument('--output-dir', default='diagnostics/TAUR_vs_UAD')
    parser.add_argument('--reuse', action='store_true', help='Reuse only provenance-verified feature caches')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('opts', nargs=argparse.REMAINDER, help='Shared YACS overrides; put them last')
    extract(parser.parse_args())
