import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from loss.taur_rank_loss import TAURRankingLoss, rank_weight
from .taur_grad import (adapter_rank_gradients, merge_adapter_gradients, synchronize_gradients,
                        direction_gradient_diagnostics, gradient_contribution, step_with_scaler)


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    return loss, ids, g_cent


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = torch.device('cuda', local_rank) if cfg.MODEL.DEVICE == 'cuda' else torch.device('cpu')
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(device)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == 'cuda' else None,
                output_device=local_rank if device.type == 'cuda' else None,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler(enabled=device.type == 'cuda')
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    if is_main:
        logger.info('Feature dimension: {}; identities: {}'.format(feat_dim, num_classes))
        logger.info('WRT stable masked softmax: %s', cfg.MODEL.WRT_STABLE)
    if use_aux_modules:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=cfg.MODEL.GPD_MOMENTUM,
            device=device,
        )

    group_size = cfg.DATALOADER.NUM_INSTANCE
    taur_on = cfg.TAUR.ENABLED
    if taur_on:
        if 'pca' not in loss_type or 'gpd' not in loss_type:
            raise ValueError('UAD+TAUR requires ProCA and GPD; use base+pca+gpd')
        rank_criterion = TAURRankingLoss(cfg.TAUR.MARGIN, cfg.TAUR.TEMPERATURE)
        adapter_params = tuple(model_meta.adapter_parameters())
        if is_main:
            logger.info('TAUR Adapter parameters: {}; ranking candidates are local to each rank'.format(
                sum(p.numel() for p in adapter_params)))
    total_steps = 0
    successful_steps, skipped_steps = 0, 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        sampler = train_loader.batch_sampler.sampler
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        rank_lambda = rank_weight(epoch, cfg.TAUR) if taur_on else 0.0
        direction_meter = {d: AverageMeter() for d in TAURRankingLoss.DIRECTIONS}
        direction_counts = {d: dict(queries=0, possible_queries=0, positive_pairs=0,
                                    negative_pairs=0, positive_camera_sum=0)
                            for d in TAURRankingLoss.DIRECTIONS}
        rank_meter = AverageMeter()
        epoch_success, epoch_skipped = 0, 0

        n_iter = 0
        for n_iter, batch in enumerate(train_loader):
            imgs, vid, camids = batch[:3]
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)
            if taur_on:
                modids, sampleids = (t.to(device) for t in batch[3:])
                all_camids = torch.cat(camids)

            # TAUR's separately computed gradients are reduced explicitly once.
            # no_sync must wrap BOTH the forward and backward, including autograd.grad.
            sync_context = model.no_sync() if dist_on and rank_lambda > 0 else nullcontext()
            if dist_on and rank_lambda > 0:
                # Repeated no_sync forwards can skip DDP buffer broadcasts.
                # Retain ordinary UAD rank-0 BN-buffer semantics explicitly.
                for buffer in model_meta.buffers():
                    dist.broadcast(buffer, src=0)
            with sync_context:
                with amp.autocast(enabled=device.type == 'cuda'):
                    cls_score, global_feat, feat = model(imgs, target, camids,
                                                         modality_ids=modids if taur_on else None)

                    loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                    meter_ls[0].update(il.item(), target_rep.shape[0])
                    meter_ls[1].update(tl.item(), target_rep.shape[0])

                    if use_aux_modules:
                        loss_pca, ids, g_cent = PCA(
                            global_feat,
                            target,
                            num_modalities=num_modalities,
                            group_size=group_size,
                        )
                        meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                        loss_gpd = GPD(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=0.03,
                        )
                        meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                        if 'pca' in loss_type:
                            loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                        if 'gpd' in loss_type:
                            loss += loss_gpd

                rank_grads = None
                diagnose = taur_on and rank_lambda > 0 and cfg.TAUR.LOG_DIRECTION_GRADS and (
                    (n_iter + 1) % log_period == 0)
                if taur_on:
                    rank_loss, direction_losses, stats = rank_criterion(
                        global_feat, target_rep, all_camids, modids, sampleids)
                    for direction in rank_criterion.DIRECTIONS:
                        direction_meter[direction].update(direction_losses[direction].item())
                        for key, count in stats[direction].items():
                            direction_counts[direction][key] += count
                    rank_meter.update(rank_loss.item())
                    if rank_lambda > 0:
                        if diagnose:
                            norms = direction_gradient_diagnostics(
                                direction_losses, adapter_params, scaler, rank_lambda, ddp=dist_on)
                            rank_norms = [None] * dist.get_world_size() if dist_on else [norms]
                            if dist_on:
                                dist.all_gather_object(rank_norms, norms)
                            if is_main:
                                logger.info('TAUR AMP-safe direction gradients by rank '
                                            '(0.25*lambda; local and DDP-mean norms): %s', rank_norms)
                        rank_grads = adapter_rank_gradients(rank_loss, adapter_params, scaler, rank_lambda)
                scaler.scale(loss).backward()
                if rank_grads is not None:
                    if diagnose:
                        contribution = gradient_contribution(adapter_params, rank_grads, scaler, ddp=dist_on)
                        if is_main:
                            logger.info('TAUR mean Adapter gradient contribution before clipping: %s', contribution)
                    merge_adapter_gradients(adapter_params, rank_grads)
            if dist_on and rank_lambda > 0:
                synchronize_gradients(model)
            scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            updated, scale_before, scale_after = step_with_scaler(optimizer, scaler)
            successful_steps += int(updated)
            skipped_steps += int(not updated)
            epoch_success += int(updated)
            epoch_skipped += int(not updated)
            if is_main and not updated:
                logger.warning('AMP skipped optimizer step: epoch=%s iteration=%s scale=%s->%s',
                               epoch, n_iter + 1, scale_before, scale_after)

            if use_aux_modules:
                with torch.no_grad():
                    proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

            total_steps += 1
            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            if device.type == 'cuda':
                torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] UAD Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
                logger.info('Rank Loss: %.6f, weighted Rank: %.6f, lambda: %.5f, '
                            'gradient norm before clipping: %.6g, AMP scale: %.6g',
                            rank_meter.avg, rank_lambda * rank_meter.avg, rank_lambda,
                            total_grad_norm.item(), scale_after)

        if taur_on:
            summary = {d: dict(direction_counts[d], loss=direction_meter[d].avg)
                       for d in rank_criterion.DIRECTIONS}
            summaries = [None] * dist.get_world_size() if dist_on else [summary]
            if dist_on:
                dist.all_gather_object(summaries, summary)
            if is_main:
                for report_rank, report in enumerate(summaries):
                    for direction, counts in report.items():
                        nq = max(1, counts['queries'])
                        logger.info('TAUR epoch=%s rank=%s %s L=%.6f lambda=%.5f '
                                    'valid_q=%s/%s pos/q=%.2f neg/q=%.2f positive_cameras/q=%.2f',
                                    epoch, report_rank, direction, counts['loss'], rank_lambda,
                                    counts['queries'], counts['possible_queries'],
                                    counts['positive_pairs'] / nq, counts['negative_pairs'] / nq,
                                    counts['positive_camera_sum'] / nq)
        if is_main:
            logger.info('Actual epoch batches=%s optimizer updates=%s AMP skips=%s; cumulative '
                        'attempts=%s updates=%s skips=%s images/rank/step=%s',
                        n_iter + 1, epoch_success, epoch_skipped, total_steps,
                        successful_steps, skipped_steps, target_rep.numel())
            if hasattr(sampler, 'last_stats'):
                logger.info('Camera PKM global epoch coverage: %s', sampler.last_stats)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )

        if epoch % eval_period == 0:
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode, modality_ids=torch.as_tensor(modid, device=device))
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = torch.device(cfg.MODEL.DEVICE)
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode, modality_ids=torch.as_tensor(modid, device=device))
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
