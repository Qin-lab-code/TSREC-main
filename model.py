import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo_matrix
import time
import random
from numba import jit
import scipy.sparse as sp
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = np.sum(final, 0)/(self.layers+1)
        return item_embeddings


class SessConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(SessConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.w_sess = {}
        for i in range(self.layers):
            self.w_sess['weight_sess%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb = trans_to_cuda(self.w_sess['weight_sess%d' % (i)])(session_emb)
            session_emb = torch.mm(DA, session_emb)
            session.append(F.normalize(session_emb, p=2, dim=-1))
        sess = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        session_emb = torch.sum(sess, 0)/(self.layers+1)
        return session_emb

class GraphConv(Module):
    def __init__(self, n_users, n_items, layers, batch_size, interaction_matrix, emb_size=100):
        super(GraphConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        self.n_users = n_users
        self.n_items = n_items
        self.interaction_matrix = interaction_matrix
        self.drop_rate = 0.1
        self.norm_adj_matrix = trans_to_cuda(self.get_norm_adj_mat())
        self.sub_mat = {}
        for layer_idx in range(self.layers):
            self.sub_mat['sub_mat_1%d' % layer_idx] = trans_to_cuda(self.get_norm_adj_mat(is_drop=True))
            self.sub_mat['sub_mat_2%d' % layer_idx] = trans_to_cuda(self.get_norm_adj_mat(is_drop=True))

    def get_norm_adj_mat(self, is_drop=False):

        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter = self.interaction_matrix
        row = inter.row
        col = inter.col
        nnz = inter.nnz

        if is_drop:
            keep_idx = random.sample(list(range(nnz)), int(nnz * (1 - self.drop_rate)))
            row = row[keep_idx]
            col = col[keep_idx]
        ratings = np.ones_like(row, dtype=np.float32)

        data_dict = dict(zip(zip(row, col + self.n_users), ratings))
        data_dict.update(dict(zip(zip(col + self.n_users, row), ratings)))
        A._update(data_dict)
        # norm adj matrix0
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, ego_embeddings):
        v1_embeddings = ego_embeddings
        v2_embeddings = ego_embeddings
        all_v1_embeddings = [v1_embeddings]
        all_v2_embeddings = [v2_embeddings]


        for layer_idx in range(self.layers):
            v1_embeddings = torch.sparse.mm(self.sub_mat['sub_mat_1%d' % layer_idx], v1_embeddings)
            all_v1_embeddings += [v1_embeddings]
        all_v1_embeddings = torch.stack(all_v1_embeddings, 1)
        all_v1_embeddings = torch.mean(all_v1_embeddings, dim=1)
        s1_user_embeddings, s1_item_embeddings = torch.split(all_v1_embeddings,
                                                                       [self.n_users, self.n_items])
        # v2 - view
        for layer_idx in range(self.layers):
            v2_embeddings = torch.sparse.mm(self.sub_mat['sub_mat_2%d' % layer_idx], v2_embeddings)
            all_v2_embeddings += [v2_embeddings]
        all_v2_embeddings = torch.stack(all_v2_embeddings, 1)
        all_v2_embeddings = torch.mean(all_v2_embeddings, dim=1)
        s2_user_embeddings, s2_item_embeddings = torch.split(all_v2_embeddings,
                                                                       [self.n_users, self.n_items])
        return s1_user_embeddings, s1_item_embeddings, s2_user_embeddings, s2_item_embeddings

class CrossCompressUnit(nn.Module):

    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        self.fc_vv = trans_to_cuda(nn.Linear(dim, 1, bias=True))
        self.fc_ev = trans_to_cuda(nn.Linear(dim, 1, bias=True))
        self.fc_ve = trans_to_cuda(nn.Linear(dim, 1, bias=True))
        self.fc_ee = trans_to_cuda(nn.Linear(dim, 1, bias=True))

    def forward(self, inputs):
        v, e = inputs
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = torch.unsqueeze(v, 2)
        e = torch.unsqueeze(e, 1)
        # [batch_size, dim, dim]
        c_matrix = torch.matmul(v, e)
        c_matrix_transpose = c_matrix.permute(0, 2, 1)
        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.contiguous().view(-1, self.dim)
        # [batch_size, dim]
        v_intermediate = self.fc_vv(c_matrix) + self.fc_ev(c_matrix_transpose)
        e_intermediate = self.fc_ve(c_matrix) + self.fc_ee(c_matrix_transpose)
        v_output = v_intermediate.view(-1, self.dim)
        e_output = e_intermediate.view(-1, self.dim)

        return v_output, e_output


class TSREC(Module):
    def __init__(self, adjacency, n_node, n_users, interaction_matrix, lr, layers, l2, beta, lam, eps, dataset,
                 emb_size=100, batch_size=100):
        super(TSREC, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_users = n_users
        self.n_items = n_node
        self.n_node = n_node
        self.dataset = dataset
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.drop_rate = 0.1
        self.ssl_temp = 0.2
        self.ssl_reg = 0.1
        self.ssl_reg1 = 1
        self.reg_weight = 1e-05
        self.K = 10
        self.w_k = 10
        self.num = 5000
        self.adjacency = adjacency
        self.interaction_matrix = interaction_matrix
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset == 'retailrocket':
            self.pos_len = 300
        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_size)
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(self.layers)
        self.SessGraph = SessConv(self.layers, self.batch_size)
        self.ConvGraph = GraphConv(self.n_users, self.n_items, self.layers, self.batch_size,
                                   self.interaction_matrix, self.emb_size)
        self.norm_adj_matrix = trans_to_cuda(self.get_norm_adj_mat())

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.linear_zero = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.linear_one = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.linear_two = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.linear_three = nn.Linear(self.emb_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.adv_item = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.adv_sess = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        # self.adv_item = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        # self.adv_sess = torch.zeros(self.n_node, self.emb_size).requires_grad_(True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.L = 1
        self.cc_unit = {}
        for i_cnt in range(self.L):
            self.cc_unit['unit%d' % i_cnt] = trans_to_cuda(CrossCompressUnit(self.emb_size))
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = trans_to_cuda(nn.Linear(self.emb_size, self.emb_size, bias=False))
        self.w_user = {}
        for i in range(self.layers):
            self.w_user['weight_user%d' % (i)] = trans_to_cuda(nn.Linear(self.emb_size, self.emb_size, bias=False))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_norm_adj_mat(self, is_drop=False):

        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter = self.interaction_matrix
        row = inter.row
        col = inter.col
        nnz = inter.nnz

        if is_drop:
            keep_idx = random.sample(list(range(nnz)), int(nnz * (1 - self.drop_rate)))
            row = row[keep_idx]
            col = col[keep_idx]
        ratings = np.ones_like(row, dtype=np.float32)

        data_dict = dict(zip(zip(row, col + self.n_users), ratings))
        data_dict.update(dict(zip(zip(col + self.n_users, row), ratings)))
        A._update(data_dict)
        # norm adj matrix0
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select



    def caculate_user_loss(self, user, pos_item, neg_item):

        # ssl loss
        user_emb1 = self.s1_user_embeddings[user]
        user_emb2 = self.s2_user_embeddings[user]

        item_emb1 = self.s1_item_embeddings[pos_item]
        item_emb2 = self.s2_item_embeddings[pos_item]

        merge_emb1 = torch.cat([user_emb1, item_emb1], dim=0)
        merge_emb2 = torch.cat([user_emb2, item_emb2], dim=0)

        # cosine similarity
        norm_merge_emb1 = F.normalize(merge_emb1, p=2, dim=1)
        norm_merge_emb2 = F.normalize(merge_emb2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_merge_emb1, norm_merge_emb2), dim=1)
        ttl_score = torch.matmul(norm_merge_emb1, norm_merge_emb2.T)

        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).mean()
        ssl_loss = self.ssl_reg * ssl_loss

        # calculate BPR Loss
        u_embeddings = self.user_main_embeddings[user]
        pos_embeddings = self.item_main_embeddings[pos_item]
        neg_embeddings = self.item_main_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log(torch.sigmoid(pos_scores-neg_scores)).mean()

        # u_ego_embeddings = self.user_embedding(user)
        # pos_ego_embeddings = self.item_embedding(pos_item)
        # neg_ego_embeddings = self.item_embedding(neg_item)

        # reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        # print("mf", mf_loss)
        # print("ssl", ssl_loss)
        if mf_loss > 1000 or ssl_loss > 1000:
            print("wrong")
            print("mf", mf_loss)
            print("ssl", ssl_loss)
            print("user",user)
            print("pos_item",pos_item )
            print("neg_item",neg_item )
            print("pos_scores", pos_scores)
            print("neg_scores",neg_scores )
            mf_loss = 0.001
        loss = mf_loss + ssl_loss
        return loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, train, diff_mask, user, item, neg_item, all_item):

        values = self.adjacency.data
        indices = np.vstack((self.adjacency.row, self.adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = self.adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = self.embedding.weight
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = [ego_embeddings]
        for i in range(self.layers):
            # item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            # ego_embeddings = trans_to_cuda(self.w_user['weight_user%d' % (i)])(ego_embeddings)
            ego_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings)
            if i < self.L:
                user_main_embedding, item_main_embedding = torch.split(ego_embeddings, [self.n_users, self.n_items])
                item_main_embedding, item_embeddings = self.cc_unit['unit%d' % (i)](
                    [item_main_embedding, item_embeddings])
                ego_embeddings = torch.cat([user_main_embedding, item_main_embedding], dim=0)
            final.append(item_embeddings)
            all_embeddings.append(ego_embeddings)
        item_embeddings_i = np.sum(final, 0) / (self.layers + 1)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_main_embeddings, item_main_embeddings = torch.split(all_embeddings,
                                                                 [self.n_users, self.n_items])
        # item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)

        if train:
            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)

            sess_emb_s = self.SessGraph(self.embedding.weight, D, A, session_item, session_len)
            sess_emb_s = self.w_k * F.normalize(sess_emb_s, dim=-1, p=2)

            # ssl loss
            user_emb1 = sess_emb_i
            user_emb2 = sess_emb_s

            item_emb = item_embeddings_i[item]

            merge_emb1 = torch.cat([user_emb1, item_emb], dim=0)
            merge_emb2 = torch.cat([user_emb2, item_emb], dim=0)

            # cosine similarity
            norm_merge_emb1 = F.normalize(merge_emb1, p=2, dim=1)
            norm_merge_emb2 = F.normalize(merge_emb2, p=2, dim=1)
            pos_score = torch.sum(torch.mul(norm_merge_emb1, norm_merge_emb2), dim=1)
            ttl_score = torch.matmul(norm_merge_emb1, norm_merge_emb2.T)

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.log(pos_score / ttl_score).mean()

            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)

            if loss_item > 1000 or ssl_loss > 1000:
                print("wrong")
                print("loss_item", loss_item)
                print("ssl", ssl_loss)
            return self.ssl_reg1 * ssl_loss + loss_item, scores_item

        else:

            ego_embeddings = self.get_ego_embeddings()
            self.user_main_embeddings, self.item_main_embeddings = user_main_embeddings, item_main_embeddings
            self.s1_user_embeddings, self.s1_item_embeddings, self.s2_user_embeddings, self.s2_item_embeddings = self.ConvGraph(
                ego_embeddings)
            loss = self.caculate_user_loss(user, item, neg_item)
            return loss, 0




def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask, user, item, neg_item, all_item = data.get_slice(i)
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    if train:
        A_hat, D_hat = data.get_overlap(session_item)
        A_hat = trans_to_cuda(torch.Tensor(A_hat))
        D_hat = trans_to_cuda(torch.Tensor(D_hat))
    else:
        A_hat, D_hat = None, None
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    user = trans_to_cuda(torch.Tensor(user).long())
    item = trans_to_cuda(torch.Tensor(item).long())
    all_item = trans_to_cuda(torch.Tensor(all_item).long())
    neg_item = trans_to_cuda(torch.Tensor(neg_item).long())
    loss, scores_item = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, epoch,tar, train, diff_mask, user, item, neg_item, all_item)
    return tar, scores_item, loss

@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    graph_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i in tqdm(slices):
        model.zero_grad()
        tar, scores_item, loss = forward(model, i, train_data, epoch, train=False)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  # 更改为梯度
        model.optimizer.step()
        graph_loss += loss.item()
    print('\tLoss1:\t%.3f' % graph_loss)
    for i in tqdm(slices):
        model.zero_grad()
        tar, scores_item, loss = forward(model, i, train_data, epoch, train=True)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  # 更改为梯度
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 20, 50]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        tar, scores_item, loss = forward(model, i, test_data, epoch, train=True)
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(50, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                prediction_list = prediction.tolist()
                epsilon = 0.1 ** 10
                DCG = 0
                IDCG = 0
                for j in range(K):
                    if prediction_list[j] == target:
                        DCG += 1 / math.log2(j + 2)
                for j in range(min(1, K)):
                    IDCG += 1 / math.log2(j + 2)
                metrics['ndcg%d' % K].append(DCG / max(IDCG, epsilon))
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


