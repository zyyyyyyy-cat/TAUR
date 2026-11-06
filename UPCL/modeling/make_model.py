import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU
import torch
from utils.cluster import ClusterMemory

class Uni_modal(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(Uni_modal, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        self.temp = cfg.MODEL.TEMP
        self.momentum = cfg.MODEL.MOMENTUM
        self.use_cluster = cfg.MODEL.USE_CLUSTER
        self.use_spe = cfg.MODEL.USE_SPE
        self.use_unbiased =cfg.MODEL.USE_UNBIASED
        
        self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_r.apply(weights_init_classifier)
        self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_r.bias.requires_grad_(False)
        self.bottleneck_r.apply(weights_init_kaiming)
        self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_n.apply(weights_init_classifier)
        self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_n.bias.requires_grad_(False)
        self.bottleneck_n.apply(weights_init_kaiming)
        self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier_t.apply(weights_init_classifier)
        self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_t.bias.requires_grad_(False)
        if self.use_cluster:
            self.memory_RGB = ClusterMemory(self.feat_dim, self.num_classes, temp=self.temp,
                                momentum=self.momentum, use_hard= False).cuda()
            self.memory_NI = ClusterMemory(self.feat_dim, self.num_classes, temp=self.temp,
                                momentum=self.momentum, use_hard=False).cuda()
            self.memory_TI = ClusterMemory(self.feat_dim, self.num_classes, temp=self.temp,
                                momentum=self.momentum, use_hard=False).cuda()
            
        if self.use_spe:
            self.memory_species_RGB = ClusterMemory(self.feat_dim , 2, temp=self.temp,
                               momentum=self.momentum, use_hard= False).cuda()
            self.memory_species_NI = ClusterMemory(self.feat_dim , 2, temp=self.temp,
                            momentum=self.momentum, use_hard=False).cuda()
            self.memory_species_TI = ClusterMemory(self.feat_dim , 2, temp=self.temp,
                            momentum=self.momentum, use_hard=False).cuda()
        if self.use_unbiased:
            self.memory_unbiased = ClusterMemory(self.feat_dim, self.num_classes, temp=self.temp,
                                    momentum=self.momentum, use_hard=False).cuda()
         
        
    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        if self.training:
            RGB = x['RGB']
            RGB1 = x['RGB1']
            NI = x['NI']
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            RGB1_cash, RGB1_global = self.BACKBONE(RGB1, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
            
            
            RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
            RGB1_ori_score = self.classifier_r(self.bottleneck_r(RGB1_global))
            NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
            TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))
            
            
            ori_RNT = torch.cat([self.bottleneck_r(RGB_global),self.bottleneck_r(RGB1_global), self.bottleneck_n(NI_global), self.bottleneck_t(TI_global)], dim=0)  
            ori_RNT_score = torch.cat([RGB_ori_score,RGB1_ori_score, NI_ori_score,TI_ori_score], dim=0)
            return (RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global,RGB1_ori_score, RGB1_global),(ori_RNT,ori_RNT_score)
        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
           
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            return RGB_global,NI_global,TI_global,ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = Uni_modal(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building DeMo===========')
    return model
