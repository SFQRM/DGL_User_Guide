import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
# from dgl.nn import SAGEConv
from dgl.utils import expand_as_pair


"""
    加载图数据
    （有时还需要划分训练集和测试集；尤其在链接预测上还需要构造负采样数据后+将正边和负边放在一起，形成训练和测试集），
    下面只是举例一个简单的分类模型
"""
# from tutorial_utils import load_zachery


def load_zachery():
    # g_ = dgl.graph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    # g_.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                 (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                 (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                 (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                 (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                 (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                 (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                 (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                 (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                 (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                 (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                 (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                 (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = torch.tensor(tuple(zip(*edge_list)))
    # print(src)
    # g.add_edges(src, dst)
    g_ = dgl.graph((src, dst))
    # edges are directional in DGL; make them bi-directional
    # g.add_edges(dst, src)
    g_ = dgl.graph((dst, src))
    g_.ndata['club'] = torch.eye(34)
    return g_


# ----------- 0. load graph -------------- #
g = load_zachery()
print("g:", g)

# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)   # 每个节点的嵌入维度为5
print("node_embed:", node_embed)
inputs = node_embed.weight                          # 使用嵌入权重作为节点特征
nn.init.xavier_uniform_(inputs)                     # 服从均匀分布
print("inputs:", inputs)
labels = g.ndata['club']
print("labels:", labels)
labeled_nodes = [0, 33]
print("labeled_nodes:", labeled_nodes)
print('Labels:', labels[labeled_nodes])


# ----------- 2. create model -------------- #

# build a two-layer GraphSAGE model
class SAGEConv(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# Create the model with given dimensions
# input layer dimension: 5, node embeddings
# hidden layer dimension: 16
# output layer dimension: 2, the two classes, 0 and 1
net = SAGEConv(5, 16, 2)
