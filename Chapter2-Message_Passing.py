import dgl
import torch
import torch.nn as nn
import dgl.function as fn

# """

# 2.1 内置函数和消息传递API

"""
    # 要对源节点的hu特征和目的节点的hv特征求和，然后将结果保存在边的he特征上。
"""

N = 6  # 节点数目
u, v = torch.tensor([1, 1, 1, 2, 2, 4, 0, 5, 5]), torch.tensor([2, 3, 4, 3, 5, 3, 1, 3, 4])
g = dgl.graph((u, v))                      # 生成DGL图
g.ndata["ht"] = torch.ones(N) / N  # 将每个节点的hu值初始化为1/N
# g.ndata["hv"] = g.out_degrees(g.nodes()).float()  # 将每个节点的out-degree存储为节点特征hv
weights = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1.])   # 每条边的权重
g.edata["a"] = weights
# 内置的消息传递函数
# fn.u_add_v("hu", "hv", "m")

# 内置的消息聚合函数
# fn.sum("m", "he")


def update_all_example(g):
    g.update_all(fn.u_mul_e("ht", "a", "m"),
                 fn.sum("m", "ht"))
    final_ht = g.ndata["ht"] * 2
    return final_ht


for i in range(5):
    print(update_all_example(g))
    
# print(final_ht)
# """


"""
# 2.2 编写高效的消息传递代码

N = 6  # 节点数目
u, v = torch.tensor([1, 1, 1, 2, 2, 4, 0, 5, 5]), torch.tensor([2, 3, 4, 3, 5, 3, 1, 3, 4])
u, v = u.to("cuda:0"), v.to("cuda:0")
g = dgl.graph((u, v))

g.ndata["feat"] = torch.clone(torch.ones(N) / N).to("cuda:0")  # 将每个节点的hu值初始化为1/N
print(g.ndata["feat"])

node_feat_dim = g.ndata["feat"].size()[0]
out_dim = torch.rand(2).size()[0]
print(node_feat_dim)
print(out_dim)

linear = nn.Parameter(torch.FloatTensor(size=(node_feat_dim * 2, out_dim)))

print(g.edges.src)

# def concat_message_function(edges):
    # return {"cat_feat": torch.cat([edges.src.ndata["feat"], edges.dst.ndata["feat"]], dim=1)}


# g.apply_edges(concat_message_function(g.edges()))
# g.edata["out"] = g.edata["cat_feat"] * linear

# print(g.edata["out"])
"""

"""
# 2.3 在图的一部分上进行消息传递

N = 6  # 节点数目
u, v = torch.tensor([1, 1, 1, 2, 2, 4, 0, 5, 5]), torch.tensor([2, 3, 4, 3, 5, 3, 1, 3, 4])
u, v = u.to("cuda:0"), v.to("cuda:0")
g = dgl.graph((u, v)).to("cuda:0")
print(g)

g.ndata["ht"] = torch.clone(torch.ones(N) / N).to("cuda:0")  # 将每个节点的hu值初始化为1/N
g.ndata["hv"] = g.out_degrees(g.nodes()).float()  # 将每个节点的out-degree存储为节点特征hv

print(g.ndata["ht"])
print(g.ndata["hv"])

weights = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])   # 每条边的权重
g.edata["a"] = weights.to("cuda:0")
print(g.edata["a"])


nid = [1, 3, 5]
sg = g.subgraph(nid)


def update_all_example(g):
    g.update_all(fn.u_mul_e("ht", "a", "m"),
                 fn.sum("m", "ht"))
    final_ht = g.ndata["ht"] * 2
    return final_ht


for i in range(5):
    print(update_all_example(sg))

"""
