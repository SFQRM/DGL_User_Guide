import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl

N = 10          # 节点数目
DAMP = 0.85     # 阻尼因子
K = 10          # 迭代次数
g = nx.nx.erdos_renyi_graph(N, 0.2)         # 图随机生成器，生成nx图
g = dgl.from_networkx(g)                    # 转换成DGL图
# print(g)
# g = g.to('cuda:0')

'''
plt.figure()            # 创建一个图形对象
plt.subplot(111)        # 创建一个子图,行列
nx.draw(g.to_networkx(),
        node_size=250,                      # 设置节点大小
        node_color=[[.75, .75, .75]])       # 设置节点灰度值
plt.show()
'''

g.ndata['pv'] = torch.ones(N)/N                     # 将每个节点的PageRank值初始化为1/N
g.ndata['deg'] = g.out_degrees(g.nodes()).float()   # 将每个节点的out-degree存储为节点特征

print(g.ndata['pv'])
print(g.ndata['deg'])
