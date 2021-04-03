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
from augmentations import get_aug, get_aug_fedmatch
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
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss

    
def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 10)

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
    
    dict_users_labeled = set(np.random.choice(list(all_idxs), int(len(all_idxs) * label_rate), replace=False))
        
    for i in range(num_users):
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(all_idxs, int(num_items * label_rate), replace=False))
#         all_idxs = list(set(all_idxs) - dict_users_labeled)
        dict_users_unlabeled[i] = set(np.random.choice(all_idxs, int(num_items) , replace=False))
        all_idxs = list(set(all_idxs) - dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
    return dict_users_labeled, dict_users_unlabeled

# def noniid(dataset, num_users, label_rate):

#     num_shards, num_imgs = 2 * num_users, int(len(dataset)/num_users/2)
#     idx_shard = [i for i in range(num_shards)]
#     dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
# #     print(type(labels))

#     num_items = int(len(dataset)/num_users)
#     dict_users_labeled = set()
#     pseduo_label = [i for i in range(len(dataset))] 

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#索引值
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

#     for i in range(num_users):

#         dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(list(dict_users_unlabeled[i]), int(num_items * label_rate), replace=False))
#         dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


#     return dict_users_labeled, dict_users_unlabeled

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
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(list(dict_users_unlabeled[i]), int(num_items * label_rate), replace=False))
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled


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
        transform=get_aug_fedmatch(args.dataset, True), 
        train=True, 
        **dataset_kwargs
    )
    
    if args.iid == 'iid':
        dict_users_labeled, dict_users_unlabeled = iid(dataset_train, args.num_users, args.label_rate)
    else:
        dict_users_labeled, dict_users_unlabeled = noniid(dataset_train, args.num_users, args.label_rate)
        
    train_loader_unlabeled = {}


    # define model
    model_glob = get_model('fedfixmatch', args.backbone).to(device)
    if torch.cuda.device_count() > 1: model_glob = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_glob)
    accuracy = []


    for iter in range(args.num_epochs):

        model_glob.train()
        optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01, momentum=0.5)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
        
        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),
            shuffle=True,
            **dataloader_kwargs
        )           
        
        for batch_idx, ((img, img_ema), label) in enumerate(train_loader_labeled):   
            input_var = torch.autograd.Variable(img.cuda())
            ema_input_var = torch.autograd.Variable(img_ema.cuda())
            target_var = torch.autograd.Variable(label.cuda())                
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(-1).sum()    
            ema_model_out = model_glob(ema_input_var)
            model_out = model_glob(input_var)
            if isinstance(model_out, Variable):
                logit1 = model_out
                ema_logit = ema_model_out
            else:
                assert len(model_out) == 2
                assert len(ema_model_out) == 2
                logit1, logit2 = model_out
                ema_logit, _ = ema_model_out    
                       
            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
            class_logit, cons_logit = logit1, logit1
            class_loss = class_criterion(class_logit, target_var) / minibatch_size
            ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
            pseudo_label1 = torch.softmax(model_out.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label1, dim=-1)
            mask = max_probs.ge(args.threshold_pl).float()
            Lu = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask).mean()
            loss = class_loss + Lu 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
#             batch_loss.append(loss.item())
                

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
            
            model_local = copy.deepcopy(model_glob).to(args.device)

            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled[idx]),
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )


            model_local.train()


            for i, ((img, img_ema), label) in enumerate(train_loader_unlabeled):

                input_var = torch.autograd.Variable(img.cuda())
                ema_input_var = torch.autograd.Variable(img_ema.cuda())
                target_var = torch.autograd.Variable(label.cuda())                
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                ema_model_out = model_local(ema_input_var)
                model_out = model_local(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out    

                ema_logit = Variable(ema_logit.detach().data, requires_grad=True)
                class_logit, cons_logit = logit1, logit1
#                 class_loss = class_criterion(class_logit, target_var) / minibatch_size
#                 ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
                pseudo_label1 = torch.softmax(model_out.detach_(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label1, dim=-1)
                mask = max_probs.ge(args.threshold_pl).float()
                Lu = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask).mean()
                loss = Lu 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
    #             batch_loss.append(loss.item())

            w_locals.append(copy.deepcopy(model_local.state_dict()))
#             loss_locals.append(sum(loss_local) / len(loss_local) )

            del model_local
            gc.collect()
            del train_loader_unlabeled
            gc.collect()
            torch.cuda.empty_cache()
            
            

        w_glob = FedAvg(w_locals)
        model_glob.load_state_dict(w_glob)

#         loss_avg = sum(loss_locals) / len(loss_locals)
        
        if iter%1==0:
            print('Round {:3d}, Acc {:.2f}%'.format(iter, acc))

#     f = open("./result_ablation.txt",'a')
#     f.write("fedavg-fixmatch")
#     f.write(str(args.label_rate))
#     f.write("\n") 
#     f.write(str(args.frac))
#     f.write(str(args.batch_size))
#     f.write("\n") 
#     f.write(args.dataset)
#     f.write("\n")   
#     f.write(args.iid)
#     f.write("\n")   
#     f.write(" ".join(accuracy)) 
#     f.write("\n")  
#     f.close()   

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
















