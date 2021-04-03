import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import copy

import gc
    
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, PlotLogger, knn_monitor
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from torch.utils.data import DataLoader, Dataset


import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import re
import argparse
import os
import shutil
import time
import math
import logging
import os
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        log_probs, _, _, _ = net_g(data, data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss

    
def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 100)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length) 
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # print(w_avg.keys())
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def iid(dataset, num_users, label_rate):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_labeled, dict_users_unlabeled = set(), {}
    dict_users_unlabeled_server = set()
    
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
    dict_users_unlabeled_server = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate * 5), replace=False))
    
    for i in range(num_users):
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(all_idxs, int(num_items * label_rate), replace=False))
#         all_idxs = list(set(all_idxs) - dict_users_labeled)
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled - dict_users_unlabeled_server
    return dict_users_labeled, dict_users_unlabeled_server, dict_users_unlabeled


def noniid(dataset, num_users, label_rate):

    num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        
    num_items = int(len(dataset)/num_users)
    dict_users_labeled = set()
    dict_users_unlabeled_server = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#索引值
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    dict_users_unlabeled_server = set(np.random.choice(list(idxs), int(len(idxs) * label_rate * 5), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(list(dict_users_unlabeled[i]), int(num_items * label_rate), replace=False))
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled - dict_users_unlabeled_server


    return dict_users_labeled, dict_users_unlabeled_server, dict_users_unlabeled

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (images1, images2), labels = self.dataset[self.idxs[item]]
        return (images1, images2), labels



def main(device, args):


    loss1_func = nn.CrossEntropyLoss()
    loss2_func = softmax_kl_loss

    dataset_kwargs = {
        'dataset':args.dataset,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size':args.batch_size if args.debug else None
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs = {
        'batch_size': args.batch_size*5,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataset_train =get_dataset(
        transform=get_aug(args.dataset, True), 
        train=True, 
        **dataset_kwargs
    )

    if args.iid == 'iid':
        dict_users_labeled, dict_users_unlabeled_server, dict_users_unlabeled = iid(dataset_train, args.num_users, args.label_rate)
    else:
        dict_users_labeled, dict_users_unlabeled_server, dict_users_unlabeled = noniid(dataset_train, args.num_users, args.label_rate)
    train_loader_unlabeled = {}


    # define model
    model_glob = get_model('global', args.backbone).to(device)
    if torch.cuda.device_count() > 1: model_glob = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_glob)
    

    model_local_idx = set()
    model_local_dict = {}
    accuracy = []
    lr_scheduler = {}


    for iter in range(args.num_epochs):

        model_glob.train()
        optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01, momentum=0.5)

        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),
            shuffle=True,
            **dataloader_kwargs
        )           
        train_loader_unlabeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_unlabeled_server),
            shuffle=True,
            **dataloader_unlabeled_kwargs
        )    
        train_loader = zip(train_loader_labeled, train_loader_unlabeled)
        
        for batch_idx, (data_x, data_u) in enumerate(train_loader): 
            (images1_l, images2_l), labels = data_x
            (images1_u, images2_u), _ = data_u
            
            model_glob.zero_grad()
            labels = labels.cuda()
            
            batch_size = images1_l.shape[0]
            images1 = torch.cat((images1_l, images1_u)).to(args.device)
            images2 = torch.cat((images2_l, images2_u)).to(args.device)
            
            z1_t, z2_t, z1_s, z2_s = model_glob.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            model_glob.update_moving_average( iter*1000 + batch_idx, 20000)

            loss_class = 1/2 * loss1_func(z1_t[:batch_size], labels) + 1/2 * loss1_func(z2_t[:batch_size], labels)
            
            loss_consist = 1/2 * loss2_func(z1_t, z2_s) / len(labels) + 1/2 * loss2_func(z2_t, z1_s) / len(labels)
            consistency_weight = get_current_consistency_weight(batch_idx)
            loss = loss_class# + consistency_weight * loss_consist

            loss.backward()
            optimizer.step()

        del train_loader_labeled
        gc.collect()
        torch.cuda.empty_cache()
            
        if iter%1==0:
            test_loader = torch.utils.data.DataLoader(
                dataset=get_dataset( 
                    transform=get_aug(args.dataset, False, train_classifier=False), 
                    train=False,
                    **dataset_kwargs),
                shuffle=False,
                **dataloader_kwargs
            )
            model_glob.eval()
            acc, loss_train_test_labeled = test_img(model_glob, test_loader, args)
            accuracy.append(str(acc))
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()
            
            
        w_locals, loss_locals, loss0_locals, loss2_locals = [], [], [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        for idx in idxs_users:
            
            loss_local = []
            loss0_local = []
            loss2_local = []

                
            if idx in model_local_idx:
                model_local = get_model('local', args.backbone).to(device)
                model_local.projector.load_state_dict(model_local_dict[idx][0])
                model_local.target_encoder.load_state_dict(model_local_dict[idx][1])
#                 model_local.projector.load_state_dict(torch.load('/model/'+'model1' + str(args.dataset) + str(idx)+ '.pkl'))
#                 model_local.target_encoder.load_state_dict(torch.load('/model/'+'model1' + str(args.dataset) + 'tar'+ str(idx)+ '.pkl'))
                
                model_local.backbone.load_state_dict(model_glob.backbone.state_dict())
            else:
                model_local = get_model('local', args.backbone).to(device)
                model_local.backbone.load_state_dict(model_glob.backbone.state_dict())
                model_local.target_encoder.load_state_dict(model_local.online_encoder.state_dict())
                model_local_idx = model_local_idx | set([idx])

            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled[idx]),
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )

            # define optimizer
            optimizer = get_optimizer(
                args.optimizer, model_local, 
                lr=args.base_lr*args.batch_size/256, 
                momentum=args.momentum,
                weight_decay=args.weight_decay)

            lr_scheduler = LR_Scheduler(
                optimizer,
                args.warmup_epochs, args.warmup_lr*args.batch_size/256, 
                args.num_epochs, args.base_lr*args.batch_size/256, args.final_lr*args.batch_size/256, 
                len(train_loader_unlabeled),
                constant_predictor_lr=True # see the end of section 4.2 predictor
            )
            
            model_local.train()

            for j in range(args.local_ep):
                
                for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled):

                    model_local.zero_grad()

                    batch_size = images1.shape[0]

                    loss = model_local.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))

                    loss.backward()
                    optimizer.step()

                    loss_local.append(int(loss))

                    lr = lr_scheduler.step()

                    model_local.update_moving_average()

                w_locals.append(copy.deepcopy(model_local.backbone.state_dict()))
                loss_locals.append(sum(loss_local) / len(loss_local) )
                model_local_dict[idx] = [model_local.projector.state_dict(), model_local.target_encoder.state_dict()]
    #             torch.save(model_local.projector.state_dict(), '/model/'+'model1' + str(args.dataset) + str(idx)+ '.pkl')
    #             torch.save(model_local.target_encoder.state_dict(), '/model/'+'model1' + str(args.dataset)+ 'tar' + str(idx)+ '.pkl')


            del model_local
            gc.collect()
            del train_loader_unlabeled
            gc.collect()
            torch.cuda.empty_cache()

            

        w_glob = FedAvg(w_locals)
        model_glob.backbone.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        
        if iter%1==0:
            print('Round {:3d}, Acc {:.2f}%'.format(iter, acc))
    

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
















