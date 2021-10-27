# -*- coding: utf-8 -*-
from typing import List, Any, Tuple
import torchvision
import copy
import numpy as np
import syft as sy
import random as rn
import torch
from model import IDCM_NN
from data_loader import load_datasets
from option import Argument
from train import *
import sys
import os
from utils import Logger, getLabelVarieties

sys.stdout = Logger('result.txt', sys.stdout)
sys.stderr = Logger('error.txt', sys.stderr)

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

hook = sy.TorchHook(torch)

#clients
alice = sy.VirtualWorker(hook, id = 'alice')
bob = sy.VirtualWorker(hook, id = 'bob')
cindy = sy.VirtualWorker(hook, id = 'cindy')
client_list = [alice, bob, cindy]

args = Argument()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

print('dataset: {}, server epoch: {:d}, client num: {:d}, client epoch: {:d}'.format(args.dataset, args.epochs, args.client_num, args.local_epochs))

print('...Data loading is beginning...')
data_loaders, input_data_pars = load_datasets(args.dataset_dir, args.dataset, args.batch_size)
print('...Data loading is completed...')


print('...Structure initialization is beginning...')
server_model = IDCM_NN(img_input_dim=input_data_pars['img_dim'], text_input_dim=input_data_pars['text_dim'],
                       output_dim=input_data_pars['num_class']).to(device)
params_to_update = list(server_model.parameters())

optimizers = {}#optimizers of clients
models = {}#models of clients
for client in client_list:
    client_model = copy.deepcopy(server_model)
    models[client] = client_model
    if args.use_fed_avg:
        optimizers[client] = torch.optim.SGD(models[client].parameters(), lr=args.alpha)
    else:
        optimizers[client] = torch.optim.Adam(models[client].parameters(), lr=args.lr, betas=args.betas)

print('...Structure initialization is completed...')

print('...Distributing is beginning...')
# send model
for client, model in models.items():
    models[client] = model.send(client)

# send train data
weights = []
label_varieties = [set(), set(), set()]
for i in range(len(client_list)):
    weights.append(0)
remote_dataset: Tuple[List[Any], List[Any], List[Any]] = (list(), list(), list())

# init client weights
for batch_idx, data in enumerate(data_loaders['train']):
    idx = batch_idx % len(client_list)
    imgs = data[0].send(client_list[idx])
    txts = data[1].send(client_list[idx])
    getLabelVarieties(data[2], label_varieties, idx)
    labels = data[2].send(client_list[idx])
    remote_dataset[idx].append((imgs, txts, labels))
    weights[idx] += len(data[0])
variety_sum = 0
for i in range(len(label_varieties)):
    variety_sum += len(label_varieties[i])
for j in range(len(weights)):
    weights[j] *= len(label_varieties[j])/variety_sum
print('...Distributing is completed...')


print('...Training is beginning...')
# train and evaluate
trainer = FedTrainer(remote_dataset, args, client_list, models, optimizers, server_model, data_loaders['test'], weights)
print('...Training is completed...')
client_best_score = trainer.train()
for KEY in client_best_score.keys():
    print('{}: {:.8f}'.format(KEY, client_best_score[KEY]))






