import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train,cfg):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train: 
            if not cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and not cfg.MODEL.USE_CROSS and not cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_baseline/basline.txt", mode='w')
            elif cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and not cfg.MODEL.USE_CROSS and not cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster/cluster_momentum_{}_temp_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP), mode='w')
            elif cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and cfg.MODEL.USE_CROSS and not cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_cross/cluster_momentum_{}_temp_{}_cross.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.CROSS_COE), mode='w')
            elif cfg.MODEL.USE_CLUSTER and cfg.MODEL.USE_SPE and not cfg.MODEL.USE_CROSS and not cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH  :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_spe/cluster_spe_momentum_{}_temp_{}_SPE_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.SPE_COE), mode='w')
            elif cfg.MODEL.USE_CLUSTER and cfg.MODEL.USE_SPE and  cfg.MODEL.USE_CROSS and not cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH  :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_spe_cross/cluster_spe_momentum_{}_temp_{}_SPE_cross_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.CROSS_COE), mode='w')
            elif cfg.MODEL.USE_CLUSTER and cfg.MODEL.USE_SPE and  cfg.MODEL.USE_CROSS and cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_spe_cross_unbiased/cluster_spe_momentum_{}_temp_{}_SPE_cross_{}_unbiased_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.CROSS_COE,cfg.MODEL.UNBIASED_COE), mode='w')  
            elif cfg.MODEL.USE_CLUSTER and cfg.MODEL.USE_SPE and  not cfg.MODEL.USE_CROSS and cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH   :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_spe_unbiased/cluster_spe_momentum_{}_temp_{}_SPE_unbiased_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.UNBIASED_COE), mode='w')  
            elif cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and  not cfg.MODEL.USE_CROSS and cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_unbiased/cluster_momentum_{}_temp_{}_unbiased_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.UNBIASED_COE), mode='w')     
            elif cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and cfg.MODEL.USE_CROSS and cfg.MODEL.USE_UNBIASED and not cfg.MODEL.USE_FINCH  :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_unbiased_cross/cluster_momentum_{}_temp_{}_unbiased_{}_cross_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.UNBIASED_COE,cfg.MODEL.CROSS_COE), mode='w')    
            elif cfg.MODEL.USE_CLUSTER and not cfg.MODEL.USE_SPE and cfg.MODEL.USE_CROSS and cfg.MODEL.USE_UNBIASED and cfg.MODEL.USE_FINCH  :
                fh = logging.FileHandler("/data1/zhouzhongao/Uni_Modality_Unbiased/logs_cluster_unbiased_cross_finch/cluster_momentum_{}_temp_{}_unbiased_{}_cross_{}_finch_{}.txt".format(cfg.MODEL.MOMENTUM,cfg.MODEL.TEMP,cfg.MODEL.UNBIASED_COE,cfg.MODEL.CROSS_COE,cfg.MODEL.FINCH_COE), mode='w')        

            
              
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
