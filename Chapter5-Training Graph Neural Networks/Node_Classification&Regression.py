# 构建一个2层的GNN模型
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

"""
    加载图数据
"""
# ----------- 0. 加载数据集 ----------- #
dataset = dgl.data.CiteseerGraphDataset(raw_dir="./data")
graph = dataset[0]
# print("graph:", graph)


"""
    加载节点特征和标签
"""
# ----------- 1. 加载节点特征与标签 ----------- #
node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]                     # 特征的维度：3703
n_labels = int(node_labels.max().item() + 1)            # 标签的数量：6

# print("node_features:", node_features)
# print("node_labels:", node_labels)
# print("train_mask", train_mask)
# print("valid_mask", valid_mask)
# print("test_mask", test_mask)


"""
    定义模型
    建立一个两层的GraphSAGE模型
"""


# ----------- 2. 编写神经网络模型 ----------- #
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
# print(model)


"""
    训练优化
    （包含模型参数-优化器等设置+前向传播+损失+反向传播）
"""
# ----------- 3. 设置优化器 ----------- #
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# print(optimizer)


# ----------- 4. 编写准确率计算方法 ----------- #
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


# ----------- 5. 训练 ----------- #
for epoch in range(100):
    model.train()

    # 使用所有节点(全图)进行前向传播计算
    logits = model(graph, node_features)
    # print(logits.shape)

    # 计算损失值
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])

    # 计算验证集的准确度
    acc = evaluate(model, graph, node_features, node_labels, valid_mask)

    # 进行反向传播计算
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('In epoch {}, loss: {}'.format(epoch, loss.item()))


print('Accuracy', acc)

# 如果需要的话，保存训练好的模型。本例中省略。
