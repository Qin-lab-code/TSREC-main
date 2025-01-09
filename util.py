import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from operator import itemgetter
import random


def data_masks(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess) - 1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i] - 1] = dict()
                    adj[sess[i] - 1][sess[i] - 1] = 1
                    adj[sess[i] - 1][sess[i + 1] - 1] = 1
                else:
                    if sess[i + 1] - 1 not in adj[sess[i] - 1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i] - 1][sess[i + 1] - 1] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo


def data_adj(all_sessions, n_node):
    adj = dict()
    row, col, data = [], [], []
    n_session = 0
    for num, sess in enumerate(all_sessions[0]):
        n_session = num + 1
        adj[num] = dict()
        for item in sess:
            if item - 1 not in adj[num].keys():
                adj[num][item - 1] = 1
            else:
                adj[num][item - 1] += 1
        item = all_sessions[1][num]
        if item - 1 not in adj[num].keys():
            adj[num][item - 1] = 1
        else:
            adj[num][item - 1] += 1
        for i in adj[num].keys():
            row.append(num)
            col.append(i)
            data.append(adj[num][i])
    coo = coo_matrix((data, (row, col)), shape=(n_session, n_node))
    return coo


class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        adj = data_masks(all_train, n_node)
        # # print(adj.sum(axis=0))
        self.adjacency = adj.multiply(1.0 / adj.sum(axis=0).reshape(1, -1))
        self.interaction = data_adj(data, n_node)
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle
        self.shuffled_arg = None

    def neg_sample(self, item_set, item_size):  # 前闭后闭
        item = random.randint(1, item_size - 1)
        while item in item_set:
            item = random.randint(1, item_size - 1)
        return item

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # matrix = self.dropout(matrix, 0.2)
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.shuffled_arg = shuffled_arg
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        user = self.shuffled_arg[index]
        items_adj = dict()
        for i in inp:
            for j in i:
                if j - 1 not in items_adj:
                    items_adj[j - 1] = 1
                else:
                    items_adj[j - 1] += 1
        for i in self.targets[index]:
            if i - 1 not in items_adj:
                items_adj[i - 1] = 1
            else:
                items_adj[i - 1] += 1
        item = []
        items_adj = sorted(items_adj.items(),key = lambda x:x[1],reverse = True)
        for i in items_adj:
            item.append(i[0])
        neg_item = []
        for i in range(len(user)):
            neg_item.append(self.neg_sample(item, self.n_node))
        all_item = item
        item = item[:len(user)]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        # item_set = set()
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        # item_set = list(item_set)
        # index_list = [item_set.index(a) for a in self.targets[index]-1]
        diff_mask = np.ones(shape=[100, self.n_node]) * (1 / (self.n_node - 1))
        for count, value in enumerate(self.targets[index] - 1):
            diff_mask[count][value] = 1
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask, diff_mask, user, item, neg_item, all_item
