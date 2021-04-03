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
from augmentations import get_aug, get_aug_uda, get_aug_fedmatch
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

def quantile_linear(iter, args):

    turn_point = int( (args.comu_rate * args.epochs - 0.1 * args.epochs -1.35) / 0.45 )
    if iter < args.phi_g:
        return 1.0
    elif iter > turn_point:
        return 0.1
    else:
        return 0.9 * iter / ( 2 - turn_point ) + 1 - 1.8/( 2 - turn_point )


def quantile_rectangle(iter, args):
    if iter < args.phi_g:
        return 0.0
    elif iter >= args.psi_g:
        return 0.0
    else:
        if args.comu_rate*5/3 > 1:
            return 0.99
        else:
            return args.comu_rate*args.epochs/(args.psi_g - args.phi_g)

def get_median(data, iter, args):
    if args.dataset == 'mnist':
        a = 8
    else:
        a = 33

    if len(data) < (39*a):
        data_test = data[(-10*a):]
    elif len(data) < (139*a):
        data_test = data[(30*a) : ]
    else:
        data_test = data[(-100*a):]

    data_test.sort()

    if args.ramp == 'linear':
        quantile = quantile_linear(iter, args)
        iter_place = int( (1 - quantile) * len(data_test))
    elif args.ramp == 'flat':
        quantile = quantile_flat(iter, args)
        iter_place = int( (1 - quantile) * len(data_test))
    elif args.ramp == 'rectangle':
        quantile = quantile_rectangle(iter, args)
        iter_place = int( (1 - quantile) * len(data_test)-1)
    else: 
        exit('Error: wrong ramp type!')
    return data_test[iter_place]

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length) 
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def sigmoid_rampup2(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length) 
        phase = current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

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

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch , args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr
    if args.lr_rampdown_epochs:
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
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
    
def get_current_consistency_weight(epoch):
    return sigmoid_rampup(epoch, 10)


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
    

    model_local_idx = set()
    
    user_epoch = {}
    lr_scheduler = {}
    accuracy = []
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
    if args.dataset == 'cifar' and args.iid != 'noniid_tradition':
        consistency_criterion = softmax_kl_loss
    else:
        consistency_criterion = softmax_mse_loss

    for iter in range(args.num_epochs):

        model_glob.train()
        optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.01, momentum=0.5)

        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),
            shuffle=True,
            **dataloader_kwargs
        )           
        
        for batch_idx, ((img, img_ema), label) in enumerate(train_loader_labeled):   
            
                img, img_ema, label = img.to(args.device), img_ema.to(args.device), label.to(args.device)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema, volatile=True)
                target_var = torch.autograd.Variable(label)
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
                classification_weight = 1 
                class_loss = classification_weight * class_criterion(class_logit, target_var) / minibatch_size
                ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
                consistency_weight = get_current_consistency_weight(iter)
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                loss = class_loss + consistency_loss
                optimizer.zero_grad()
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
            if idx in user_epoch.keys():
                user_epoch[idx] += 1 
            else:
                user_epoch[idx] = 1
                
            loss_local = []
            loss0_local = []
            loss2_local = []

                
            model_local = copy.deepcopy(model_glob).to(args.device)

            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled[idx]),
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )

            optimizer = torch.optim.SGD(model_local.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
            
            model_local.train()


            for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled):

                img, img_ema, label = img.to(args.device), img_ema.to(args.device), label.to(args.device)
                adjust_learning_rate(optimizer, user_epoch[idx], batch_idx, len(train_loader_unlabeled), args)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema, volatile=True)
                target_var = torch.autograd.Variable(label)
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
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1

                consistency_weight = get_current_consistency_weight(user_epoch[idx])
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                loss = consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            w_locals.append(copy.deepcopy(model_local.state_dict()))

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

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
















