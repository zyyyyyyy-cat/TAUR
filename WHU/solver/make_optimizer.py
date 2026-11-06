import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        if 'thermal_adapters.' in key:
            # Exactly 3x base LR for every Adapter parameter, including biases.
            if cfg.TAUR.ADAPTER_LR_FACTOR <= 0:
                raise ValueError('TAUR.ADAPTER_LR_FACTOR must be positive')
            lr = cfg.SOLVER.BASE_LR * cfg.TAUR.ADAPTER_LR_FACTOR

        group = {"params": [value], "lr": lr, "weight_decay": weight_decay}
        if 'thermal_adapters.' in key:
            group['taur_lr_factor'] = cfg.TAUR.ADAPTER_LR_FACTOR
        params.append(group)

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
