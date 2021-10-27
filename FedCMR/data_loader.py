# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
class CustomDataSet(Dataset):  # 继承torch.utils.data.dataset
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def ind2vec(ind, N=None):
    ind = np.asarray(ind)  # 确保类型为ndarray
    if N is None:
        N = ind.max()    # max函数返回数组中的最大值，作为右开区间的取值
    return np.arange(1, N + 1) == np.repeat(ind, N, axis=1)#返回一个由False和True组成的ndarray


def load_datasets(path, dataset_name, batch_size):
    if dataset_name =='Wikipedia':
        trainPath = os.path.join(os.getcwd(), path, dataset_name, "train")
        testPath = os.path.join(os.getcwd(), path, dataset_name, "test")
        img_train = torch.tensor(loadmat(os.path.join(trainPath, "train_imgs.mat"))['features'], dtype=torch.float32)
        img_test = torch.tensor(loadmat(os.path.join(testPath, "test_imgs.mat"))['features'], dtype=torch.float32)
        text_train = torch.tensor(loadmat(os.path.join(trainPath, "train_texts.mat"))['features'], dtype=torch.float32)
        text_test = torch.tensor(loadmat(os.path.join(testPath, "test_texts.mat"))['features'], dtype=torch.float32)
        label_train = loadmat(os.path.join(trainPath, "train_labels.mat"))['labels']
        label_test = loadmat(os.path.join(testPath, "test_labels.mat"))['labels']
        # 转为tensor
        label_train = torch.tensor(ind2vec(label_train).astype(int))
        # label_train被转为tensor，每一行说明对应的Each pair of instances属于哪一个类别，其中一行[0,0,1,0,0,0,0,0,0,0]表示属于第三类
        label_test = torch.tensor(ind2vec(label_test).astype(int))

    if dataset_name == 'Pascal':
        Path = os.path.join(os.getcwd(), path, dataset_name)
        train_number = 800
        valid_number = 100
        test_number = -100
        img = torch.tensor(loadmat(os.path.join(Path, "imgs_shuffled.mat"))['features'], dtype=torch.float32)
        img_train = img[0:train_number]
        img_val = img[train_number:test_number]
        img_test = img[test_number:]
        text = torch.tensor(loadmat(os.path.join(Path, "texts_shuffled.mat"))['features'], dtype=torch.float32)
        text_train = text[0:train_number]
        text_val = text[train_number:test_number]
        text_test = text[test_number:]
        label = torch.tensor(loadmat(os.path.join(Path, "labels_shuffled.mat"))['labels'], dtype=torch.int64)
        label = torch.tensor(ind2vec(label).astype(int))
        label_train = label[0:train_number]
        label_val = label[train_number:test_number]
        label_test = label[test_number:]

    if dataset_name == 'MIR-Flickr25K':
        Path = os.path.join(os.getcwd(), path, dataset_name)
        train_number = 16012
        test_number = -2002
        img = torch.tensor(loadmat(os.path.join(Path, "imgs.mat"))['features'], dtype=torch.float32)
        img_train = img[0:train_number]
        img_test = img[test_number:]
        text = torch.tensor(loadmat(os.path.join(Path, "texts.mat"))['features'], dtype=torch.float32)
        text_train = text[0:train_number]
        text_test = text[test_number:]
        label = torch.tensor(loadmat(os.path.join(Path, "labels.mat"))['labels'], dtype=torch.int64)
        label_train = label[0:train_number]
        label_test = label[test_number:]

    if dataset_name == 'MS-COCO':
        trainPath = os.path.join(os.getcwd(), path, dataset_name, "train")
        testPath = os.path.join(os.getcwd(), path, dataset_name, "test")
        img_train = torch.tensor(loadmat(os.path.join(trainPath, "train_imgs.mat"))['features'],
                                 dtype=torch.float32)
        img_test = torch.tensor(loadmat(os.path.join(testPath, "test_imgs.mat"))['features'], dtype=torch.float32)
        text_train = torch.tensor(loadmat(os.path.join(trainPath, "train_texts.mat"))['features'],
                                  dtype=torch.float32)
        text_test = torch.tensor(loadmat(os.path.join(testPath, "test_texts.mat"))['features'], dtype=torch.float32)
        label_train = torch.tensor(loadmat(os.path.join(trainPath, "train_labels.mat"))['labels'], dtype=torch.int64)
        label_test = torch.tensor(loadmat(os.path.join(testPath, "test_labels.mat"))['labels'], dtype=torch.int64)
        test_number = -30137
        img_test = img_test[test_number:]
        text_test = text_test[test_number:]
        label_test = label_test[test_number:]

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}

    # 初始化训练集和测试集
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    # set to ``True`` to have the data reshuffled at every epoch (default: ``False``)
    shuffle = {'train': False, 'test': False}

    # 初始化训练集和测试集的数据装载器
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}
    # 维度
    img_dim = img_train.shape[1]  # 4096
    text_dim = text_train.shape[1]  # 1024
    num_class = label_train.shape[1]

    # 输入数据对 类型：字典
    input_data_par = {}

    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train

    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test

    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par

