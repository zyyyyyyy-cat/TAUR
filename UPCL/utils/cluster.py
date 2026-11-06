import logging
import os
import sys
import os.path as osp
import time
from collections import OrderedDict
import torch
from torch.cuda import amp
import collections
from abc import ABC
import torch.nn.functional as F
from torch import nn, autograd

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def extract_features(model, train_loader, print_freq=50,flip=True,mode=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_RGB = OrderedDict()
    features_NI = OrderedDict()
    features_TI = OrderedDict()
    labels = []
    species_labels = []
    frames_list =[]
    end = time.time()
    device = "cuda"
    with torch.no_grad():
        for i, (img, vid, target_cam,_, target_view,fnames) in enumerate(train_loader): #target_view为catgory_id
        
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = torch.tensor(vid, dtype=torch.int64).to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # target_species = target_view.to(device)
            with amp.autocast(enabled=True):
                feats_RGB,feats_NI,feats_TI,_ = model(img, label=target, cam_label=target_cam, view_label=target_view)

            for fname, feat_RGB,feat_NI,feat_TI, id,species_id in zip(fnames, feats_RGB,feats_NI,feats_TI,vid,target_view):
                features_RGB[fname] =  feat_RGB
                features_NI[fname] =  feat_NI
                features_TI[fname] =  feat_TI
                labels.append(id)
                frames_list.append(fname)
                species_labels.append(int(species_id))
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features_RGB,features_NI,features_TI,labels,frames_list,species_labels

def extract_features_tsne(model, train_loader, print_freq=50,flip=True,mode=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # features_RGB = OrderedDict()
    # features_NI = OrderedDict()
    # features_TI = OrderedDict()
    labels = []
    species_labels = []
    frames_list =[]
    end = time.time()
    device = "cuda"
    with torch.no_grad():
        for i, (img, vid, target_cam,_, target_view,fnames) in enumerate(train_loader): #target_view为catgory_id
        
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = torch.tensor(vid, dtype=torch.int64).to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # target_species = target_view.to(device)
            with amp.autocast(enabled=True):
                feats_RGB,feats_NI,feats_TI,_ = model(img, label=target, cam_label=target_cam, view_label=target_view)
            if i ==0:
                features_RGB = feats_RGB
                features_NI = feats_NI
                features_TI = feats_TI
                labels = target
            else:
                features_RGB = torch.concatenate((features_RGB,feats_RGB),0)
                features_NI = torch.concatenate((features_NI,feats_NI),0)
                features_TI = torch.concatenate((features_TI,feats_TI),0)
                labels = torch.concatenate((labels,target),0)
            # for fname, feat_RGB,feat_NI,feat_TI, id,species_id in zip(fnames, feats_RGB,feats_NI,feats_TI,vid,target_view):
            #     features_RGB[fname] =  feat_RGB
            #     features_NI[fname] =  feat_NI
            #     features_TI[fname] =  feat_TI
            #     labels.append(id)
            #     frames_list.append(fname)
            #     species_labels.append(int(species_id))
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features_RGB,features_NI,features_TI,labels,frames_list,species_labels
@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss