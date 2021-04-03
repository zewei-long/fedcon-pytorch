#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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