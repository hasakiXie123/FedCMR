# -*- coding: utf-8 -*-

class Argument():
    def __init__(self):

        self.alpha = 1e-3# model parameter
        self.beta = 1e-1

        self.lr = 1e-4# the learning rate of the ADAM optimiser
        self.betas = (0.5, 0.999)# the model paramter of the ADAM optimiser
        self.mu = 0.1#0.5*paramter of FedProx
        self.loss_factor = [0.5, 0.3, 0.2]

        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 40  # server 迭代次数 10
        self.momentum = 0.5  # 冲量
        self.no_cuda = False
        self.seed = 10
        self.log_interval = 30
        self.save_model = False

        self.local_epochs = 2  # client迭代次数
        self.client_num = 3  # client个数
        self.all_clients = True  # 是否每轮所有client都参与训练
        self.frac = 1 if self.all_clients else 0.5  # 每一轮每一个client被选中参与训练的概率
        self.dataset = 'Pascal'  # 'Pascal', 'MS-COCO','Wikipedia','MIR-Flickr25K'
        self.multiLabel = False #是否使用多标签数据集
        self.dataset_dir = 'data'

        self.use_fed_avg = False


