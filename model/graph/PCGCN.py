import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss
import torch.nn.functional as F
from scipy.sparse import spmatrix

from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE,info_nce
from data.augmentor import GraphAugmentor
from data.utils import shuffle
import os
import math
import scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class PCGCN(GraphRecommender):
    def __init__(self, conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU):
        super(PCGCN, self).__init__(conf, training_set_IF, test_set,training_set_IU,test_IF,test_IU)
        args = OptionConf(self.config['PCGCN'])
        self.cl_rate = float(args['-lambda'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        aug_type = self.aug_type = int(args['-augtype'])
        self.model = PCGCN_Encoder(self.data, self.emb_size, drop_rate, n_layers, temp,self.config,aug_type)


    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb,rec_user_emb_implict, rec_item_emb_implict = model()
                user_emb, pos_item_emb, neg_item_emb,user_emb_implict, pos_item_emb_implict, neg_item_emb_implict = \
                    rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx],\
                    rec_user_emb_implict[user_idx], rec_item_emb_implict[pos_idx], rec_item_emb_implict[neg_idx]
                rec_loss=bpr_loss(user_emb,pos_item_emb,neg_item_emb)
                rec_loss_neg=bpr_loss(user_emb_implict,pos_item_emb_implict,neg_item_emb_implict)
                cl_loss=self.cl_rate*model.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  rec_loss+rec_loss_neg + l2_reg_loss(self.reg, user_emb, pos_item_emb)+l2_reg_loss(self.reg, user_emb_implict, pos_item_emb_implict) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb,self.user_emb_implict,self.item_emb_implict = self.model()
            if epoch>=0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb, self.user_emb_implict, self.item_emb_implict = self.best_user_emb, self.best_item_emb,self.best_user_emb_implict, self.best_item_emb_implict


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, self.best_user_emb_implict, self.best_item_emb_implict = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()



class PCGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp,config,aug_type):
        super(PCGCN_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.norm_adj = data.norm_adj
        self.Implict_Norm = data.norm_adj_implicit
        self.norm_adj_total=data.norm_adj_total
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_norm_adj_implict = TorchGraphInterface.convert_sparse_mat_to_tensor(self.Implict_Norm).cuda()
        self.sparse_norm_adj_total=TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj_total).cuda()
        self.config=config
        self.aug_type = aug_type



    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
            'user_emb_implict': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb_implict': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),

        })

        return embedding_dict


    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings_implict = torch.cat([self.embedding_dict['user_emb_implict'], self.embedding_dict['item_emb_implict']], 0)
        all_embeddings = [ego_embeddings]
        all_embeddings_implict = [ego_embeddings_implict]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings_implict = torch.sparse.mm(self.sparse_norm_adj_implict, ego_embeddings_implict)
            all_embeddings.append(ego_embeddings)
            all_embeddings_implict.append(ego_embeddings_implict)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        all_embeddings_implict = torch.stack(all_embeddings_implict, dim=1)
        all_embeddings_implict = torch.mean(all_embeddings_implict, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_implict, item_all_embeddings_implict = torch.split(all_embeddings_implict, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings, user_all_embeddings_implict, item_all_embeddings_implict


    def CrossReadout(self,emb,emb_implict):
        all_embeddings_throught_implicit = [emb]
        all_embeddings_implict_throught_pos = [emb_implict]
        for k in range(self.n_layers):
            emb = torch.sparse.mm(self.sparse_norm_adj_implict, emb)
            emb_implict = torch.sparse.mm(self.sparse_norm_adj_implict, emb_implict)
            all_embeddings_throught_implicit.append(emb)
            all_embeddings_implict_throught_pos.append(emb_implict)
        all_embeddings_throught_implicit = torch.stack(all_embeddings_throught_implicit, dim=1)
        all_embeddings_throught_implicit = torch.mean(all_embeddings_throught_implicit, dim=1)
        all_embeddings_implict_throught_pos = torch.stack(all_embeddings_implict_throught_pos, dim=1)
        all_embeddings_implict_throught_pos = torch.mean(all_embeddings_implict_throught_pos, dim=1)
        user_P2I, item_P2I = torch.split(all_embeddings_throught_implicit, [self.data.user_num, self.data.item_num])
        user_I2P, item_I2P = torch.split(all_embeddings_implict_throught_pos,
                                                                               [self.data.user_num, self.data.item_num])
        return user_P2I,item_P2I,user_I2P,item_I2P

    def CrossReadout2(self, emb, emb_implict):
        all_embeddings = [emb]
        all_embeddings_implict = [emb_implict]
        for k in range(self.n_layers):
            emb = torch.sparse.mm(self.sparse_norm_adj, emb)
            emb_implict = torch.sparse.mm(self.sparse_norm_adj, emb_implict)
            all_embeddings.append(emb)
            all_embeddings_implict.append(emb_implict)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        all_embeddings_implict = torch.stack(all_embeddings_implict, dim=1)
        all_embeddings_implict = torch.mean(all_embeddings_implict, dim=1)
        user, item = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        user_I, item_I = torch.split(all_embeddings_implict,
                                         [self.data.user_num, self.data.item_num])
        return user, item, user_I, item_I

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_P2I, item_P2I, user_I2P, item_I2P = self.CrossReadout(
            torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0),
            torch.cat([self.embedding_dict['user_emb_implict'], self.embedding_dict['item_emb_implict']], 0))
        user_P, item_P, user_I, item_I = self.CrossReadout2(torch.cat([user_P2I, item_P2I], 0), torch.cat(
            [user_I2P, item_I2P], 0))
        user_view1_P = user_P[u_idx]
        user_view2_I = user_I[u_idx]
        item_view1_P = item_P[i_idx]
        item_view2_I = item_I[i_idx]
        view1_P = torch.cat((user_view1_P, item_view1_P), 0)
        view2_I = torch.cat((user_view2_I, item_view2_I), 0)
        return InfoNCE(view1_P,view2_I,self.temp)



