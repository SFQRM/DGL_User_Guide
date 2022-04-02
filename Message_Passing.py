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