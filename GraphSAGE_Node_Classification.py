import numpy as np
import dgl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import pandas as pd
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv import SAGEConv


def load_zachery():
    nodes_data = pd.read_csv('data/nodes.csv')
    edges_data = pd.read_csv('data/edges.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g_ = dgl.graph((src, dst))
    club = nodes_data['Club'].to_list()
    # Convert to categorical integer values with 0 for 'Mr. Hi', 1 for 'Officer'.
    club = torch.tensor([c == 'Officer' for c in club]).long()
    # We can also convert it to one-hot encoding.
    club_onehot = F.one_hot(club)
    g_.ndata.update({'club': club, 'club_onehot': club_onehot})
    return g_


"""
    加载图数据
    （有时还需要划分训练集和测试集；尤其在链接预测上还需要构造负采样数据后+将正边和负边放在一起，形成训练和测试集），
    下面只是举例一个简单的分类模型
"""
# ----------- 0. load graph -------------- #
g = load_zachery()
print("g:", g)

"""
    节点特征和部分标签初始化
"""
# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)   # 将34个节点，嵌入到5维的向量中
# print("node_embed:", node_embed)                  # node_embed: Embedding(34, 5)
inputs = node_embed.weight
# print("inputs:\n", inputs)
nn.init.xavier_uniform_(inputs)                     # 通过网络层后输入和输出的方差相同，并服从均为分布
# print("inputs:\n", inputs)

# g.ndata["club"] = inputs
labels = g.ndata["club"]
labeled_nodes = [0, 33]
# print('Labels', labels[labeled_nodes])


"""
    模型定义
    如GraphSage2层模型定义，使用dgl内置函数。
    需要注意的是如果是消息传递模型（区别于DGL内置函数），则需要额外定义重写内置模型SAGEConv
"""
# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
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
net = GraphSAGE(5, 16, 2)


"""
    训练优化
    （包含模型参数-优化器等设置+前向传播+损失+反向传播）
"""
# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), node_embed.parameters()), lr=0.01)


# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    logits = net(g, inputs)

    # compute loss
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[labeled_nodes], labels[labeled_nodes])

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))


# ----------- 5. check results ------------------------ #
pred = torch.argmax(logits, axis=1)
print('Accuracy', (pred == labels).sum().item() / len(pred))
