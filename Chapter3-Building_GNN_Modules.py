# DGL NN模块的构造函数

import torch.nn as nn

from dgl.utils import expand_as_pair


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,                  # 输入的维度
                 out_feats,                 # 输出的维度
                 aggregator_type,           # 聚合类型
                 bias=True,
                 norm=None,                 # 特征归一化
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)   # 源节点特征输入维度和目标节点特征输入维度
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation

        # 聚合类型：mean、max_pool、lstm、gcn
        if aggregator_type not in ['mean', 'max_pool', 'lstm', 'gcn']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        if aggregator_type == 'max_pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type in ['mean', 'max_pool', 'lstm']:
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()