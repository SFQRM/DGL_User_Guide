import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl
import dgl.function as fn

N = 6  # 节点数目
DAMP = 0.85  # 阻尼因子
K = 10  # 迭代次数
# g = nx.nx.erdos_renyi_graph(N, 0.2)  # 图随机生成器，生成nx图
# g = dgl.from_networkx(g)  # 转换成DGL图
u, v = torch.tensor([1, 1, 1, 2, 2, 4, 0, 5, 5]), torch.tensor([2, 3, 4, 3, 5, 3, 1, 3, 4])
g = dgl.graph((u, v))                      # 生成DGL图
print(g)
# g = g.to('cuda:0')

plt.figure()  # 创建一个图形对象
plt.subplot(111)  # 创建一个子图,行列
nx.draw(g.to_networkx(),
        node_size=250,  # 设置节点大小
        node_color=[[.75, .75, .75]])  # 设置节点灰度值
plt.show()

g.ndata["pagerank"] = torch.ones(N) / N  # 将每个节点的PageRank值初始化为1/N
g.ndata["degree"] = g.out_degrees(g.nodes()).float()  # 将每个节点的out-degree存储为节点特征

print(g.ndata['pagerank'])
print(g.ndata['degree'])


"""
# 定义message函数，它将每个节点的PageRank值除以其out-degree，并将结果作为消息传递给它的邻居：
def pagerank_message_func(edges):
    return {"pv": edges.src["pagerank"] / edges.src["degree"]}


# 定义reduce(聚合)函数，它从mailbox中删除并聚合message，并计算其新的PageRank值：
def pagerank_reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox["pv"], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {"pv": pv}
"""

# g.register_message_func(pagerank_message_func)
# g.register_reduce_func(pagerank_reduce_func)

"""
def pagerank_naive(g):
    # Phase #1: 沿所有边缘发送消息。
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Phase #2: 接收消息以计算新的PageRank值。
    for v in g.nodes():
        g.recv(v)


pagerank_naive(g)
print(g.ndata["pagerank"])
"""


def paggrank_build(g):
    g.ndata["pagerank"] = g.ndata["pagerank"] / g.ndata["degree"]
    g.update_all(message_func=fn.copy_src(src="pagerank", out="m"),
                 reduce_func=fn.sum(msg="m", out = "m_sum"))
    g.ndata["pagerank"] = (1 - DAMP) / N + DAMP * g.ndata["m_sum"]


for i in range(5):
    paggrank_build(g)
    print(g.ndata["pagerank"])

