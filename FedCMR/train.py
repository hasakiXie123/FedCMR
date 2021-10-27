import torch
import utils
import numpy as np
from evaluate import fx_calc_map_label, fx_calc_map_multilabel_k
import math
import heapq

class FedTrainer(object):
    def __init__(self, remote_dataset, args, client_list, models, optimizers, server_model, test_data_loader, weights):
        self.remote_dataset = remote_dataset
        self.args = args
        self.client_list = client_list
        self.models = models#List containing the model_ptr sent to selected/each client
        self.server_model = server_model
        self.optimizers = optimizers#List containing the optimizers of selected/each client
        self.test_data_loader = test_data_loader
        self.weights = weights

    def train(self):
        client_num = self.args.client_num  # number of clients
        client_best_score = {}
        if client_num != len(self.client_list):
            print("Wrong client num!")

        i = 0  # iteration number

        if self.args.all_clients:
            print("...Aggregation over all clients...")

        while i < self.args.epochs:

            m = max(int(self.args.frac * client_num), 1)  # 本轮参与训练的client数量
            client_selected_idxs = np.random.choice(range(client_num), m, replace=False)
            client_selected = []
            for idx in client_selected_idxs:
                    client_selected.append(self.client_list[idx])

            p = [math.ceil(0.8 * len(self.remote_dataset[i])) for i in range(client_num)]  # 本轮参与训练的client数量
            client_data_idxs = [{x for x in range(len(self.remote_dataset[i]))} for i in range(client_num)]
            #80%
            client_data_selected_idxs = [np.random.choice(range(len(self.remote_dataset[i])), p[i], replace=False) for i in range(client_num)]
            #20%
            client_data_local_train_idxs = [client_data_idxs[i] - set(client_data_selected_idxs[i]) for i in range(client_num)]

            loss = torch.zeros(1, client_num)#client loss 列表
            ori_weights = {}
            ori_bias = {}
            # client本地的训练
            for j in range(client_num):
                # Update each client
                client_loss_sum = 0.0
                self.models[self.client_list[j]].get()
                for name, param in self.models[self.client_list[j]].named_parameters():
                    if name == 'linearLayer.weight':
                        ori_weights[self.client_list[j]] = param.data
                    if name == 'linearLayer.bias':
                        ori_bias[self.client_list[j]] = param.data
                self.models[self.client_list[j]].send(self.client_list[j])
                for k in range(self.args.local_epochs):
                    self.models[self.client_list[j]].train()
                    for data_index in client_data_selected_idxs[j]:
                        imgs, txts, labels = self.remote_dataset[j][data_index]

                        with torch.set_grad_enabled(True):
                            if torch.cuda.is_available():
                                imgs = imgs.cuda()
                                txts = txts.cuda()
                                labels = labels.cuda()
                            self.optimizers[self.client_list[j]].zero_grad()
                            view1_feature, view2_feature, view1_predict, view2_predict = self.models[self.client_list[j]](imgs, txts)
                            client_loss = utils.calc_loss(view1_feature, view2_feature, view1_predict,
                                        view2_predict, labels, labels, self.args.alpha, self.args.beta)
                            client_loss.backward()
                            self.optimizers[self.client_list[j]].step()

                            client_loss_sum += client_loss.get()
                    print("client {:d} epoch: {:d}th done".format(j, k))
                loss[0][j] = client_loss_sum
            with torch.no_grad():
                client_params_weights = []
                client_params_bias = []
                for client in self.models.keys():
                    client_model = self.models[client]
                    client_model.get()
                    weights = {}
                    bias = {}
                    for name, param in client_model.named_parameters():
                        if name == 'linearLayer.weight':
                            weights[name] = param
                        if name == 'linearLayer.bias':
                            bias[name] = param
                    client_params_weights.append(weights)
                    client_params_bias.append(bias)

                w_glob = utils.parameterCooperation(client_params_weights, self.weights, loss)
                b_glob = utils.parameterCooperation(client_params_bias, self.weights, loss)

                param_dict = dict(self.server_model.named_parameters())
                for name in param_dict.keys():
                    if name == 'linearLayer.weight':
                        param_dict[name].set_(w_glob[name])
                    if name == 'linearLayer.bias':
                        param_dict[name].set_(b_glob[name])

                for client in self.models.keys():
                    client_model = self.models[client]
                    client_param_dict = dict(client_model.named_parameters())
                    for name in client_param_dict.keys():
                        if name == 'linearLayer.weight':
                            w = utils.calc(ori_weights[client], client_param_dict[name], w_glob[name])
                            client_param_dict[name].set_(w)
                        if name == 'linearLayer.bias':
                            b = utils.calc(ori_bias[client], client_param_dict[name], b_glob[name])
                            client_param_dict[name].set_(b)
            for client, client_model in self.models.items():
                self.models[client] = client_model.send(client)

            for j_local in range(client_num):
                for k_local in range(self.args.local_epochs):
                    self.models[self.client_list[j_local]].train()
                    for data_index in client_data_local_train_idxs[j_local]:
                        imgs, txts, labels = self.remote_dataset[j_local][data_index]

                        with torch.set_grad_enabled(True):
                            if torch.cuda.is_available():
                                imgs = imgs.cuda()
                                txts = txts.cuda()
                                labels = labels.cuda()
                            self.optimizers[self.client_list[j_local]].zero_grad()
                            view1_feature, view2_feature, view1_predict, view2_predict = self.models[self.client_list[j_local]](imgs, txts)
                            client_loss = utils.calc_loss(view1_feature, view2_feature, view1_predict,
                                                      view2_predict, labels, labels, self.args.alpha, self.args.beta)
                            client_loss.backward()
                            self.optimizers[self.client_list[j_local]].step()

                with torch.no_grad():
                    self.models[self.client_list[j_local]].get()
                    self.models[self.client_list[j_local]].eval()
                    t_imgs, t_txts, t_labels = [], [], []
                    for imgs, txts, labels in self.test_data_loader:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.cuda()
                        t_view1_feature, t_view2_feature, _, _ = self.models[self.client_list[j_local]](imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())

                if self.args.multiLabel:
                    t_imgs = np.concatenate(t_imgs)  # 起到将t_imgs 转为ndarray的作用？
                    t_txts = np.concatenate(t_txts)
                    t_labels = np.concatenate(t_labels)  # 按列
                    img2text = fx_calc_map_multilabel_k(t_imgs, t_txts, t_labels)
                    txt2img = fx_calc_map_multilabel_k(t_txts, t_imgs, t_labels)
                else:
                    t_imgs = np.concatenate(t_imgs)  # 起到将t_imgs 转为ndarray的作用？
                    t_txts = np.concatenate(t_txts)
                    t_labels = np.concatenate(t_labels).argmax(1)  # 按列
                    img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)  # img to text
                    txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)  # text to img

                avg_score = (img2text + txt2img) / 2.0
                Img2Txt_KEY = "client" + str(j_local) + "Img2Txt"
                Txt2Img_KEY = "client" + str(j_local) + "Txt2Img"
                Best_Score_KEY = "client" + str(j_local)+ "Best_Score"
                if Img2Txt_KEY not in client_best_score.keys():
                    client_best_score[Img2Txt_KEY] = img2text
                if Txt2Img_KEY not in client_best_score.keys():
                    client_best_score[Txt2Img_KEY] = txt2img
                if Best_Score_KEY not in client_best_score.keys():
                    client_best_score[Best_Score_KEY] = avg_score
                if avg_score > client_best_score[Best_Score_KEY]:
                    client_best_score[Img2Txt_KEY] = img2text
                    client_best_score[Txt2Img_KEY] = txt2img
                    client_best_score[Best_Score_KEY] = avg_score
                print('client {:d}： Img2Txt: {:.8f}  Txt2Img: {:.8f}'.format(j_local, img2text, txt2img))

            for client, client_model in self.models.items():
                self.models[client] = client_model.send(client)

            if i % 10 == 0:
                print("iteration " + str(i))

            i += 1
            for KEY in client_best_score.keys():
                print('{}: {:.8f}'.format(KEY, client_best_score[KEY]))

        return client_best_score