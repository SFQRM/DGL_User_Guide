import dgl
import torch as th

"""
# 1.2 图、节点和边

'''
对于多个节点，DGL使用一个一维的整型张量来保存图的点ID， DGL称之为”节点张量”。
为了指代多条边，DGL使用一个包含2个节点张量的元组(U, V)，其中，用(U[i], V[i])指代一条U[i]到V[i]的边。
'''

# edges 0->1, 0->2, 0->3, 1->3
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))

print(g)

# 获取节点的ID
print(g.nodes())
# 获取边的对应端点
print(g.edges())

# 对于无向的图，用户需要为每条边都创建两个方向的边。
bg = dgl.to_bidirected(g)

print(bg.edges())

"""

# 1.3 节点和边的特征

u, v = th.tensor([0, 0, 1, 5]), th.tensor([1, 2, 2, 0])     # 6个节点，4条边
g = dgl.graph((u, v))

g.ndata['x'] = th.ones(g.num_nodes(), 3)        # 为每个节点创建长度为3的节点特征
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # 标量整型特征

# 不同名称的特征可以具有不同形状
g.ndata['y'] = th.randn(g.num_nodes(), 5)
print(g.ndata['x'][1])                      # 获取节点1的特征
print(g.edata['x'][th.tensor([0, 3])])      # 获取边0和3的特征

# 边 0->1, 0->2, 0->3, 1->3
edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
weights = th.tensor([0.1, 0.6, 0.9, 0.7])   # 每条边的权重
g = dgl.graph(edges)
g.edata['w'] = weights                      # 将其命名为 'w'
print(g.edata['w'])