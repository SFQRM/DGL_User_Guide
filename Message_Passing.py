import dgl
import torch
import torch.nn as nn

"""
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
u, v = u.to('cuda:0'), v.to('cuda:0')
cuda_g = dgl.graph((u, v))
# print(cuda_g)
cuda_g.ndata['hu'] = th.randn(4, 4).to('cuda:0')
cuda_g.ndata['hv'] = th.randn(4, 4).to('cuda:0')
print(cuda_g.ndata['hu'])
print(cuda_g.ndata['hv'])

# cuda_g.ndata['he'] = dgl.function.u_add_v('hu', 'hv', 'he')
# print(cuda_g.ndata['he'])
"""

# 2.1 内置函数和消息传递API

"""
    要对源节点的hu特征和目的节点的hv特征求和，然后将结果保存在边的he特征上。
"""
import dgl.function as fn

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
"""

# 2.2 编写高效的消息传递代码

u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))

node_feat_dim, out_dim = range(4)

linear = nn.Parameter(torch.FloatTensor(size=(node_feat_dim * 2, out_dim)))


def concat_message_function(edges):
    return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']], dim=1)}


g.apply_edges(concat_message_function)
g.edata['out'] = g.edata['cat_feat'] @ linear
"""