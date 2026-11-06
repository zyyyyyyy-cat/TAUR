import torch
import torch.nn as nn
from collections.abc import Mapping
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup


def read_reid_checkpoint(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, Mapping):
        for key in ('state_dict', 'model'):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping):
        raise ValueError('Expected a ReID state_dict checkpoint')
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError('Checkpoint state_dict must contain named tensors')
        name = key[7:] if key.startswith('module.') else key
        if name in result:
            raise ValueError('Duplicate checkpoint key after prefix removal: ' + name)
        result[name] = value
    return result


def load_reid_state(model, state, finetune=False):
    expected = model.state_dict()
    adapters = {k for k in expected if 'thermal_adapters.' in k}
    saved_adapters = {k for k in state if 'thermal_adapters.' in k}
    initialize_adapters = finetune and bool(adapters) and not saved_adapters
    if not initialize_adapters and adapters != saved_adapters:
        raise ValueError('Checkpoint/config Adapter mismatch; use matching TAUR settings')
    # Outer classifier and unused ImageNet head are unnecessary for retrieval.
    ignored = lambda key: key.startswith(('classifier.', 'base.fc.'))
    required = {k for k in expected if not ignored(k)}
    if initialize_adapters:
        required -= adapters
    missing = required - state.keys()
    unexpected = {k for k in state if k not in expected and not ignored(k)}
    wrong_shapes = [k for k in required & state.keys() if expected[k].shape != state[k].shape]
    if missing or unexpected or wrong_shapes:
        raise ValueError('Invalid ReID checkpoint: missing={}, unexpected={}, shape_mismatch={}'.format(
            sorted(missing), sorted(unexpected), sorted(wrong_shapes)))
    # Check every shape before changing any model value; no partial load on error.
    selected = {k: v for k, v in state.items() if k in required}
    if finetune:
        selected.update({k: v for k, v in state.items() if k.startswith('classifier.')
                         and k in expected and expected[k].shape == v.shape})
    model.load_state_dict(selected, strict=False)
    if initialize_adapters:
        print('Initializing TAUR Adapters from their existing initialization; shared UAD weights loaded')

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.taur_enabled = cfg.TAUR.ENABLED
        if self.taur_enabled:
            if list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']:
                raise ValueError('TAUR requires WHU modality order RGB, IR, Thermal (IDs 1,2,3)')
            if self.neck_feat != 'before' or cfg.TEST.FEAT_NORM != 'yes' or cfg.TEST.RE_RANKING:
                raise ValueError('TAUR requires normalized pre-BN retrieval without re-ranking')
            if cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW:
                raise ValueError('TAUR uses the UAD backbone without SIE')
            if cfg.TAUR.RANK_WEIGHT < 0 or cfg.TAUR.RAMP_END_EPOCH <= cfg.TAUR.WARMUP_EPOCHS:
                raise ValueError('Invalid TAUR rank weight schedule')
            self.base.enable_thermal_adapters(cfg.TAUR.LAYERS, cfg.TAUR.BOTTLENECK, cfg.TAUR.RHO)

    def adapter_parameters(self):
        return self.base.thermal_adapters.parameters()
        
    def forward(self, x=None, label=None, camids=None, mode=0, modality_ids=None):
        if mode==0:
            x = torch.cat(list(x), dim=0)
            global_feat = self.base(x, modal_label=modality_ids)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, feat
        else:
            global_feat = self.base(x, modal_label=modality_ids)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        load_reid_state(self, read_reid_checkpoint(trained_path))
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        load_reid_state(self, read_reid_checkpoint(model_path), finetune=True)
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
