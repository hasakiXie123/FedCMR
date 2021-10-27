from model import IDCM_NN
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import sys
import torch
import numpy as np
import math
import copy
import random

class Logger(object):
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def parameterCooperation(parameters, weights, loss):
    K = len(parameters)  # the number of clients
    p_avg = {}
    p_weights = copy.deepcopy(weights)
    p_loss = torch.zeros(1, K)
    for i in range(K):
        p_loss[0][i] = loss[0][i]
    factor = torch.zeros(1, K)
    softMax = nn.Softmax(dim=1)
    wl_factor = [0.2, 0.8]  # weights : loss

    p_loss = p_loss.detach().numpy()

    weightSum = 0.0
    for weight in p_weights:
        weightSum += weight
    for i in range(K):
        p_weights[i] = p_weights[i] / weightSum
    lossMean = np.mean(p_loss)#10 *
    for i in range(K):
        x = p_loss[0][i]/lossMean
        try:
            p_loss[0][i] = math.exp(-math.exp(x))
        except OverflowError:
            print(x)
            sys.exit(1)

    for i in range(K):
        f = p_weights[i] + 100.0*p_loss[0][i]#wl_factor[0]*
        factor[0][i] = f
    factor = softMax(factor)
    for netName in parameters[0].keys():
        p_avg[netName] = torch.mul(input=parameters[0][netName], other=factor[0][0])
        for j in range(1, K):
            torch.add(input=p_avg[netName], other=torch.mul(input=parameters[j][netName], other=factor[0][j]),
                      out=p_avg[netName])
    return p_avg


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
    term1 = ((view1_predict-labels_1.float())**2).sum(1).sqrt().mean() + ((view2_predict-labels_2.float())**2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    return im_loss


def getLabelVarieties(sample_labels, label_varieties, idx):
    for i in range(sample_labels.shape[0]):
        for j in range(sample_labels.shape[1]):
            if sample_labels[i][j] == 1:
                label_varieties[idx].add(j)

def calc(ori_param, client_param, glob_param):

    x = torch.sub(client_param, ori_param)
    beta = -0.5
    x = beta * x
    param = torch.add(glob_param, x)
    return param


