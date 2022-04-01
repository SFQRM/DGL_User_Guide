import dgl
import torch as th
import scipy as sp
import networkx as nx

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
"""

"""
# 1.4 从外部源创建图

# 从外部库创建图
spmat = sp.sparse.rand(100, 100, density=0.05)  # 5%非零项的稀疏矩阵
dgl.from_scipy(spmat)                           # 来自Scipy
print(dgl.from_scipy(spmat))

nx_g = nx.path_graph(5)                         # 一条链路0-1-2-3-4，有向图
dgl.from_networkx(nx_g)                         # 来自NetworkX
print(dgl.from_networkx(nx_g))
nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])  # 有向图
dgl.from_networkx(nxg)
print(dgl.from_networkx(nxg))

# 从磁盘加载图
# Load .CSV: https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb
"""

"""
# 1.5 异构图

# 创建一个具有3种节点类型和3种边类型的异构图
# 一个异构图由一系列子图构成，一个子图对应一种关系。每个关系由一个字符串三元组定义(源节点类型,边类型,目标节点类型)
graph_data = {
    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)

u, v = th.tensor([0, 0, 1, 5]), th.tensor([1, 2, 2, 0])     # 6个节点，4条边
# 一个同构图
dgl.heterograph({('node_type', 'edge_type', 'node_type'): (u, v)})
# 一个二分图
dgl.heterograph({('source_type', 'edge_type', 'destination_type'): (u, v)})
"""

# 1.6 在GPU上使用DGLGraph

u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
# print(g)
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
# print(g.ndata['x'])
# print("当前运行设备：", g.device)     # CPU

print("cuda是否可用：", th.cuda.is_available())       # 查看cuda是否可用
print("cuda数量：", th.cuda.device_count())       # 查看cuda数量
print("GPU名称：", th.cuda.get_device_name())    # 查看gpu名字
print("当前设备索引号：", th.cuda.current_device())     # 当前设备索引
print("cuda版本：", th.version.cuda)

cuda_g = g.to('cuda:0')         # 接受来自后端框架的任何设备对象
print(cuda_g.device)
print("pytroch版本：", th.__version__)
