import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP,R1_mAP_eval_unified,R1_mAP_unified
from torch.cuda import amp
import torch.distributed as dist
from utils.cluster import extract_features,generate_cluster_features
import torch.nn.functional as F
from layers.itc_loss import compute_itc
from utils.finch import FINCH,cluster_proto
from utils.focal_loss import cross_task_focal_loss
import numpy as np
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_normal,
             val_loader_201,
             val_loader_100,
             val_loader_msvr,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query_201,num_query_100,num_query_msvr,local_rank,human_id_train,vehicle_id_train):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("Uni_Modal.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator_201 = R1_mAP_eval_unified(num_query_201, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_100 = R1_mAP_eval_unified(num_query_100, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_msvr = R1_mAP_unified(num_query_msvr, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0,'epoch':0}
    if cfg.MODEL.USE_CLUSTER:
        features_RGB,features_NI,features_TI,labels,frames_list,species_labels = extract_features(model,train_loader_normal, print_freq=50,mode=1)
        features_RGB = torch.cat([features_RGB[f].unsqueeze(0) for f in sorted(frames_list)], 0)
        features_NI = torch.cat([features_NI[f].unsqueeze(0) for f in sorted(frames_list)], 0)
        features_TI = torch.cat([features_TI[f].unsqueeze(0) for f in sorted(frames_list)], 0)
        
        cluster_features_RGB = generate_cluster_features(labels, features_RGB)
        cluster_features_NI = generate_cluster_features(labels, features_NI)
        cluster_features_TI = generate_cluster_features(labels, features_NI)

        model.memory_RGB.features = F.normalize(cluster_features_RGB, dim=1).cuda()
        model.memory_NI.features = F.normalize(cluster_features_NI, dim=1).cuda()
        model.memory_TI.features = F.normalize(cluster_features_TI, dim=1).cuda()
        if cfg.MODEL.USE_UNBIASED:
            model.memory_unbiased.features = F.normalize((cluster_features_RGB + cluster_features_NI+cluster_features_TI)/3, dim=1).cuda()
            loss_mse = nn.MSELoss()
    if cfg.MODEL.USE_SPE:
        cluster_species_RGB = generate_cluster_features(species_labels, features_RGB)
        cluster_species_NI = generate_cluster_features(species_labels, features_NI)
        cluster_species_TI = generate_cluster_features(species_labels, features_NI)
    
        model.memory_species_RGB.features = F.normalize(cluster_species_RGB, dim=1).cuda()
        model.memory_species_NI.features = F.normalize(cluster_species_NI, dim=1).cuda()
        model.memory_species_TI.features = F.normalize(cluster_species_TI, dim=1).cuda()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            # img = {'RGB': img['RGB'].to(device),
            #        'NI': img['NI'].to(device),
            #        'TI': img['TI'].to(device)}
            img = {'RGB': img['RGB'].to(device),
                   'RGB1': img['RGB1'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output,output_cross = model(img, label=target, cam_label=target_cam, view_label=target_view)
                loss = 0
                
                for i in range(0, len(output), 2):
                    loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                    loss = loss + loss_tmp
                
                
                target_concat = torch.cat((target, target, target, target), 0)
                target_cam_concat = torch.cat((target_cam, target_cam, target_cam, target_cam), 0)
                for i in range(0, len(output_cross), 2):
                    loss_tmp_concat = loss_fn(score=output_cross[i], feat=output_cross[i + 1], target= target_concat, target_cam=target_cam_concat)
                    loss = loss + loss_tmp_concat
                if cfg.MODEL.USE_CLUSTER:
                    loss_cluster = model.memory_RGB(output[1],target) + model.memory_NI(output[3],target) +model.memory_TI(output[5],target) 
                    # loss_cluster = model.memory_RGB(output[1],target) + model.memory_RGB(output[3],target) +model.memory_NI(output[5],target) +model.memory_TI(output[7],target) 
                    loss = loss + loss_cluster
                if cfg.MODEL.USE_SPE:
                    loss_species = model.memory_species_RGB(output[1],target_view) + model.memory_species_NI(output[3],target_view) +model.memory_species_TI(output[5],target_view)
                    loss = loss + cfg.MODEL.SPE_COE *loss_species
                if cfg.MODEL.USE_CROSS:
                    loss_cross =  compute_itc(output[1],output[3],cfg.MODEL.TEMP)+ compute_itc(output[1],output[5],cfg.MODEL.TEMP)+ compute_itc(output[3],output[5],cfg.MODEL.TEMP)
                    loss = loss + cfg.MODEL.CROSS_COE * loss_cross
                if cfg.MODEL.USE_UNBIASED:
                    loss_unbiased = model.memory_unbiased((output[1]+output[3]+output[5])/3,target)
                    loss_unbiasd =  cfg.MODEL.UNBIASED_COE * loss_unbiased
                    loss = loss + loss_unbiasd
                if cfg.MODEL.USE_FINCH:
                    #分别对人和车的prototype进行聚类
                    proto_list_human = model.memory_unbiased.features[list(human_id_train)].cpu().numpy()
                    proto_list_vehicle = model.memory_unbiased.features[list(vehicle_id_train)].cpu().numpy()
                    c_human, num_clust_human, req_c_human = FINCH(proto_list_human, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=False)
                    c_vehicle, num_clust_vehicle, req_c_vehicle = FINCH(proto_list_vehicle, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=False)
                    cluster_proto_human = cluster_proto(c_human,proto_list_human)
                    cluster_proto_vehicle = cluster_proto(c_vehicle,proto_list_vehicle)
                    proto_human = torch.mean(torch.stack(cluster_proto_human),dim=0)
                    proto_vehicle = torch.mean(torch.stack(cluster_proto_vehicle),dim=0)
                    proto_cat = torch.cat([proto_human,proto_vehicle],dim=0)
                    proto = proto_cat[list(target_view)].cuda()
                    loss_finch = loss_mse(output[1],proto) + loss_mse(output[3],proto) + loss_mse(output[5],proto)
                    loss_finch =  cfg.MODEL.FINCH_COE * loss_finch
                    loss = loss + loss_finch
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))



        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            mAP_R2N_RGBNT201,cmc_R2N_RGBNT201,mAP_R2T_RGBNT201,cmc_R2T_RGBNT201,mAP_N2R_RGBNT201,cmc_N2R_RGBNT201,mAP_N2T_RGBNT201,cmc_N2T_RGBNT201,mAP_T2R_RGBNT201,cmc_T2R_RGBNT201,mAP_T2N_RGBNT201,cmc_T2N_RGBNT201,mAP_fused_RGBNT201,cmc_fused_RGBNT201 = training_neat_eval_unified(cfg, model, val_loader_201, device, evaluator_201, epoch, logger,
                                              return_pattern=3,dataset_name='RGBNT201')
                
            mAP_avg_201 = (mAP_R2N_RGBNT201 + mAP_R2T_RGBNT201 + mAP_N2R_RGBNT201 + mAP_N2T_RGBNT201 + mAP_T2R_RGBNT201 +mAP_T2N_RGBNT201 +mAP_fused_RGBNT201) / 7
            Rank1_avg_201 = (cmc_R2N_RGBNT201[0] + cmc_R2T_RGBNT201[0] + cmc_N2R_RGBNT201[0] + cmc_N2T_RGBNT201[0] + cmc_T2R_RGBNT201[0] + cmc_T2N_RGBNT201[0]+ cmc_fused_RGBNT201[0]) /7
            Rank5_avg_201 = (cmc_R2N_RGBNT201[4] + cmc_R2T_RGBNT201[4] + cmc_N2R_RGBNT201[4] + cmc_N2T_RGBNT201[4] + cmc_T2R_RGBNT201[4] + cmc_T2N_RGBNT201[4]+ cmc_fused_RGBNT201[4]) /7
            Rank10_avg_201 = (cmc_R2N_RGBNT201[9] + cmc_R2T_RGBNT201[9] + cmc_N2R_RGBNT201[9] + cmc_N2T_RGBNT201[9] + cmc_T2R_RGBNT201[9] + cmc_T2N_RGBNT201[9]+ cmc_fused_RGBNT201[9]) /7
            
            mAP_R2N_RGBNT100,cmc_R2N_RGBNT100,mAP_R2T_RGBNT100,cmc_R2T_RGBNT100,mAP_N2R_RGBNT100,cmc_N2R_RGBNT100,mAP_N2T_RGBNT100,cmc_N2T_RGBNT100,mAP_T2R_RGBNT100,cmc_T2R_RGBNT100,mAP_T2N_RGBNT100,cmc_T2N_RGBNT100,mAP_fused_RGBNT100,cmc_fused_RGBNT100 = training_neat_eval_unified(cfg, model, val_loader_100, device, evaluator_100, epoch, logger,
                                            return_pattern=3,dataset_name='RGBNT100')
            mAP_avg_100 = (mAP_R2N_RGBNT100 + mAP_R2T_RGBNT100+ mAP_N2R_RGBNT100 + mAP_N2T_RGBNT100 + mAP_T2R_RGBNT100 +mAP_T2N_RGBNT100+mAP_fused_RGBNT100) /7
            Rank1_avg_100 = (cmc_R2N_RGBNT100[0] + cmc_R2T_RGBNT100[0] + cmc_N2R_RGBNT100[0] + cmc_N2T_RGBNT100[0] + cmc_T2R_RGBNT100[0] + cmc_T2N_RGBNT100[0]+cmc_fused_RGBNT100[0]) /7
            Rank5_avg_100 = (cmc_R2N_RGBNT100[4] + cmc_R2T_RGBNT100[4] + cmc_N2R_RGBNT100[4] + cmc_N2T_RGBNT100[4] + cmc_T2R_RGBNT100[4] + cmc_T2N_RGBNT100[4]+ cmc_fused_RGBNT100[4]) /7
            Rank10_avg_100 = (cmc_R2N_RGBNT100[9] + cmc_R2T_RGBNT100[9] + cmc_N2R_RGBNT100[9] + cmc_N2T_RGBNT100[9] + cmc_T2R_RGBNT100[9] + cmc_T2N_RGBNT100[9]+ cmc_fused_RGBNT100[9]) /7
            
            mAP = (mAP_avg_201 + mAP_avg_100 )/2
            
            mAP_R2N_MSVR310,cmc_R2N_MSVR310,mAP_R2T_MSVR310,cmc_R2T_MSVR310,mAP_N2R_MSVR310,cmc_N2R_MSVR310,mAP_N2T_MSVR310,cmc_N2T_MSVR310,mAP_T2R_MSVR310,cmc_T2R_MSVR310,mAP_T2N_MSVR310,cmc_T2N_MSVR310,mAP_fused_MSVR310,cmc_fused_MSVR310= training_neat_eval_unified(cfg, model, val_loader_msvr, device, evaluator_msvr, epoch, logger,
                                              return_pattern=3,dataset_name='MSVR310')
            mAP_avg_MSVR310 = (mAP_R2N_MSVR310 + mAP_R2T_MSVR310+ mAP_N2R_MSVR310 + mAP_N2T_MSVR310 + mAP_T2R_MSVR310 +mAP_T2N_MSVR310 +mAP_fused_MSVR310 ) /7
            Rank1_avg_MSVR310 = (cmc_R2N_MSVR310[0] + cmc_R2T_MSVR310[0] + cmc_N2R_MSVR310[0] + cmc_N2T_MSVR310[0] + cmc_T2R_MSVR310[0] + cmc_T2N_MSVR310[0]+cmc_fused_MSVR310[0]) /7
            if mAP >= best_index['mAP']:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                best_index['mAP'] = mAP
                best_index['Rank-1'] = (Rank1_avg_201 + Rank1_avg_100)/2
                best_index['Rank-5'] = (Rank5_avg_201 + Rank5_avg_100)/2
                best_index['Rank-10'] =(Rank10_avg_201 + Rank10_avg_100)/2
                best_index['epoch'] = epoch
            logger.info("~" * 50)
            logger.info("Best mAP: {:6.2f}".format(best_index['mAP']))
            logger.info("Best Rank-1: {:6.2f}".format(best_index['Rank-1']))
            logger.info("Best Rank-5: {:6.2f}".format(best_index['Rank-5']))
            logger.info("Best Rank-10: {:6.2f}".format(best_index['Rank-10']))
            logger.info("Best epoch: {}".format(best_index['epoch']))
            logger.info("~" * 50)

            
            
                



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, return_pattern=1):
    device = "cuda"
    logger = logging.getLogger("DeMo.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern, img_path=imgpath)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1):
    evaluator.reset()
    model.eval()
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc
def Harmony_mean(data):
    if len(data) == 0:
        raise ValueError("输入数据不能为空")
    
    # 检查数据中是否有零或负数
    if any(x <= 0 for x in data):
        raise ValueError("所有数据必须是正数")
    
    # 计算倒数的平均值
    reciprocal_sum = sum(1/x for x in data)
    n = len(data)
    
    # 计算调和平均数
    return n / reciprocal_sum
def training_neat_eval_unified(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator_unified, epoch, logger, return_pattern=1,dataset_name='RGBNT201'):
    evaluator_unified.reset()
    model.eval()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat_RGB,feat_NI,feat_TI,feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern)
            if dataset_name == "MSVR310":
                evaluator_unified.update((feat_RGB,feat_NI,feat_TI,feat, vid, camid, scenceids, _))
            else:
                evaluator_unified.update((feat_RGB,feat_NI,feat_TI,feat, vid, camid, _))
    cmc_R2N,mAP_R2N,cmc_R2T, mAP_R2T, cmc_N2R, mAP_N2R, cmc_N2T, mAP_N2T, cmc_T2R, mAP_T2R, cmc_T2N, mAP_T2N, cmc_fused, mAP_fused = evaluator_unified.compute()
    Har_mAP = Harmony_mean([mAP_R2N,mAP_R2T,mAP_N2R,mAP_N2T,mAP_T2R,mAP_T2N,mAP_fused])
    Har_rank1 = Harmony_mean([cmc_R2N[0],cmc_R2T[0],cmc_N2R[0],cmc_N2T[0],cmc_T2R[0],cmc_T2N[0],cmc_fused[0]])
    Har_rank5 = Harmony_mean([cmc_R2N[4],cmc_R2T[4],cmc_N2R[4],cmc_N2T[4],cmc_T2R[4],cmc_T2N[4],cmc_fused[4]])
    Har_rank10 = Harmony_mean([cmc_R2N[9],cmc_R2T[9],cmc_N2R[9],cmc_N2T[9],cmc_T2R[9],cmc_T2N[9],cmc_fused[9]])
    logger.info("  ---------------------------------------------------")
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("  ****{}******".format(dataset_name))
    logger.info("    Mode   | # mAP  | #Rank-1| #Rank-5| #Rank-10")
    logger.info("  rgb->ni  | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_R2N*100, cmc_R2N[0]*100, cmc_R2N[4]*100, cmc_R2N[9]*100))
    logger.info("  rgb->ti  | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_R2T*100, cmc_R2T[0]*100, cmc_R2T[4]*100, cmc_R2T[9]*100))
    logger.info("  ni->rgb  | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_N2R*100, cmc_N2R[0]*100, cmc_N2R[4]*100, cmc_N2R[9]*100))
    logger.info("  ni->ti   | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_N2T*100, cmc_N2T[0]*100, cmc_N2T[4]*100, cmc_N2T[9]*100))
    logger.info("  ti->rgb  | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_T2R*100, cmc_T2R[0]*100, cmc_T2R[4]*100, cmc_T2R[9]*100))
    logger.info("  ti->ni   | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_T2N*100, cmc_T2N[0]*100, cmc_T2N[4]*100, cmc_T2N[9]*100))
    logger.info("  RNT->RNT | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(mAP_fused*100, cmc_fused[0]*100, cmc_fused[4]*100, cmc_fused[9]*100))
    logger.info("  Har_mean | {:6.2f} | {:6.2f} | {:6.2f} | {:6.2f}".format(Har_mAP*100, Har_rank1*100, Har_rank5*100, Har_rank10*100))
    logger.info("  -------------------------------------------------")
    
    return mAP_R2N*100,cmc_R2N*100,mAP_R2T*100,cmc_R2T*100,mAP_N2R*100,cmc_N2R*100,mAP_N2T*100,cmc_N2T*100,mAP_T2R*100,cmc_T2R*100,mAP_T2N*100,cmc_T2N*100,mAP_fused*100,cmc_fused*100