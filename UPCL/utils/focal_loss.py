import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_task_focal_loss(inputs_a, inputs_b, alpha=1.0, gamma=4.0):

    pt_a = torch.exp(-inputs_a)
    pt_b = torch.exp(-inputs_b)

    eps = 0.000000001


    wt_a = ((pt_b + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)
    wt_b = ((pt_a + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)


    f_loss_a = alpha * (1 + wt_a) ** gamma * inputs_a
    f_loss_b = alpha * (1 + wt_b) ** gamma * inputs_b

    loss = torch.mean(f_loss_a) + torch.mean(f_loss_b)

    return loss