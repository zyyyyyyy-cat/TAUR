# Existing-checkpoint diagnostics (V1/V2/V3)

These tools never create an optimizer, compute training gradients, or update model parameters.
The only change to production code is a default-True, nonpersistent `adapter_enabled`
attribute in `TransReID`; False bypasses both existing adapters without deleting their
checkpoint keys. Normal single-GPU and DDP training use the original enabled path.
`train.py`, `test.py`, `processor.py`, losses, sampler, optimizer, prototype updates,
distributed synchronization and `utils/metrics.py` are unchanged by this addition.

## Run on the server

From the `WHU` directory, using the existing original UAD and TAUR epoch-120 weights:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/diagnose_taur_checkpoint.py \
  --uad-weight /path/to/original_UAD/transformer_120.pth \
  --taur-weight logs/TAUR_1000_DDP_v2/transformer_120.pth \
  --output-dir diagnostics/TAUR_vs_UAD \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.DATASET_DIR WHU-MARS
```

Use `python`, one process and one GPU. Do not use torchrun. The UAD weight must be
the original baseline being compared, not a newly trained sampler/WRT control.
Default configs: `configs/UAD.yml` and `configs/TAUR.yml`; specify
`--uad-config` and `--taur-config` if the saved models used different architecture
settings. Shared YACS overrides go last, after all `--...` options.
No ImageNet weight is needed; checkpoint retrieval weights are loaded by the existing
strict loader. UAD has no adapters. TAUR-on/off use the identical checkpoint.

`--extract-only` stops after the three exports. Rerun with `--reuse` to reuse verified
feature caches: checkpoint SHA256, effective config, file SHA256, actual dataset
metadata and order must all match. Missing caches are extracted. An unverified or
stale existing cache is rejected rather than silently reused. Dataset image contents
are not individually hashed: keep the dataset fixed when reusing caches. Existing
third-party feature exports are not trusted automatically; these tools require their
own extraction manifest and metadata schema.

Offline analysis needs no CUDA, dataset or checkpoints once extraction is complete:

```bash
python tools/analyze_taur_ranking.py \
  --output-dir diagnostics/TAUR_vs_UAD \
  --chunk-size 32 --bootstrap 1000
```

`--chunk-size` bounds temporary query-by-gallery matrices, not gallery length.
`--bootstrap 0` disables bootstrap; `--no-plot` omits the optional PNG. Matplotlib is
optional. CPU sorting and detailed full-gallery statistics can take time.

## Exact extraction and evaluation contract

- Existing validation loaders, resize/normalization and original query/gallery splits.
  Train loaders are constructed by the original factory but never iterated.
- Evaluation runs FP32, `eval()` and `torch.inference_mode()`; model state hashes before
  and after each extraction must agree. No stochastic training augmentation or AMP.
- Raw 768D pre-BN CLS is saved. Analysis L2-normalizes it as original evaluation does.
- Full paths come from the loader's underlying dataset rows. Every batch's PID,
  camera, true per-image modid and filename is checked against those rows. The mode
  loop index is only the existing model-forward selector, not metadata.
- NPZ rows contain all query modalities followed by all gallery modalities, matching
  original evaluator splitting. `index` is the original concatenated index within
  query or gallery; `modality_index` is the index within that modality/split.
- Modalities are RGB=1, IR/NIR=2, Thermal/TIR=3. Dataset zero-based cameras 0-4
  are Ground (`G`), 5-6 Aerial (`A`), exactly as the current official metric module.
  Unexpected camera/modality IDs are rejected. Platform never substitutes for camera.
- All same-camera gallery images are excluded, including different identities.
  Queries with no remaining positive image are excluded for each protocol. The valid
  query index sequence must be identical across all three states for that protocol.
- Full gallery sorting: normalized FP32 squared Euclidean distance and
  `torch.argsort`, as the existing evaluator. AP uses every positive; INP is
  `num_valid_pos / last_pos_rank`. No top-K, reranking, or BN-neck features.
- Pairwise metadata and raw RGB/NIR on/off feature equivalence are asserted before
  any comparison CSV is produced. Feature tolerance: atol=1e-6, rtol=1e-5.
- Equal scores follow Torch's default sorting order. Tie counts are recorded; tiny
  FP32 rounding or different hardware can change near-tie ranks. V2 negative counts
  follow actual sorted positions; V3 `C_img` and `C_id` count strictly higher cosine
  similarity, so they can differ from rank counts on ties. Margin sign is not used
  as a substitute for the actual computed Rank-1.

## Outputs

| File | Contents |
|---|---|
| `uad_features.npz` | UAD raw features and metadata |
| `taur_on_features.npz` | TAUR enabled features and metadata |
| `taur_off_features.npz` | Same TAUR checkpoint, both adapters bypassed |
| `diagnostic_manifest.json` | Checkpoint/feature hashes, effective configs, versions, counts and evaluation settings |
| `checkpoint_comparison.csv` | Overall, nine modality pairs, TIR→ALL; valid counts, mAP/mINP, Rank-1/5/10 |
| `adapter_feature_effect.csv` | Query/gallery per-modality 1−cos statistics, raw max difference and tolerances |
| `query_rank_transitions.csv` | Three-state per-query ranking metrics, three pairwise deltas and Rank-1 transitions |
| `ranking_summary.csv` | Repair/damage counts, first-rank/AP categories, tail changes, margin/error-count patterns and bootstrap intervals |
| `tir_ranking_mechanism.csv` | Per-query margins, strict image/identity outranking counts, hardest negatives and per-modality valid positive/negative counts |
| `tir_platform_breakdown.csv` | TIR G→G, A→A, G→A, A→G with actual camera filtering |
| `tir_mixed_gallery_summary.csv` | TIR→ALL wrong top-1 modality fractions and explicitly reported denominators |
| `rank_ap_changes.png` | Optional ALL→ALL first-rank versus AP deltas; TIR queries, RGB/NIR queries with TIR strongest positive, and other queries |

CSV aggregate metric and confidence-interval values are percentages/percentage points.
Per-query `AP`/`INP` and their deltas are fractions; ranks are 1-based.
Comparisons are TAUR-on minus UAD, TAUR-on minus TAUR-off, and TAUR-off minus UAD.
Lower rank/error counts are better; higher AP/INP/margins are better.
Empty protocols have zero valid queries and NaN metrics, not fabricated zeros.
Queries without negatives have valid AP/INP but undefined margins (NaN).

In the mixed-gallery summary, the RGB/NIR/TIR fractions divide by queries with an
incorrect overall top-1; the error-query fraction divides by all valid TIR queries.
`most_similar_negative_*` in the mechanism CSV describes the strongest negative even
when the query is correctly retrieved; it must not be counted as an incorrect top-1.
The supplementary `TIR_best_positive_gallery` summary group includes queries whose
strongest positive is TIR in either compared state. `cross_platform_best_positive`
similarly includes either state's strongest positive on the other platform. These
are descriptive groups, not a replacement for the four fixed platform protocols.

Bootstrap resamples query PID clusters with replacement, retaining all queries per
sampled identity and the original query-weighted metric means. It estimates query
sampling variation at fixed checkpoints; it does not establish across-training-seed
stability. B-off is an inference intervention, not an independently trained backbone
control. Do not attribute all UAD→TAUR-off differences solely to the adapter: sampler,
WRT and joint training also differ. Verify reproduction of known checkpoint metrics
before drawing mechanism conclusions; this tooling does not assume the numerical
declines quoted in the guidance are universally true.

## Local validation limits

Regression tests cover both-layer bypass, unchanged state_dict/weights and RGB/NIR,
known complete rankings including a positive beyond rank 50, all-camera filtering,
ties, invalid queries, negative-free queries, exact agreement with the original
metric on synthetic full rankings, three-state pairing, cache integrity, clustered
bootstrap, synthetic export and interrupted extraction/reuse. Full real-data results
require the server dataset and existing checkpoints; none are generated locally.
