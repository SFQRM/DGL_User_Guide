"""
# 4.1 DGLDataset类
from dgl.data import DGLDataset


#     DGLDataset是处理、导入和保存dgl.data中定义的图数据集的基类。
#     DGLDataset的目的是提供一种标准且方便的方式来导入图数据。
#     用户可以存储有关数据集的图、特征、标签、掩码，以及诸如类别数、标签数等基本信息。


class MyDataset(DGLDataset):
    # 用于在DGL中自定义图数据集的模板：

    # Parameters
    # ----------
    # url : str
    #     下载原始数据集的url。
    # raw_dir : str
    #     指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    # save_dir : str
    #     处理完成的数据集的保存目录。默认：raw_dir指定的值
    # force_reload : bool
    #     是否重新导入数据集。默认：False
    # verbose : bool
    #     是否打印进度信息。


    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        pass

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        pass

    def __len__(self):
        # 数据样本的数量
        pass

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        pass

    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        pass

    def has_cache(self):
        # 检查在 `self.save_path` 中是否存有处理后的数据
        pass
"""

# 4.3 处理数据
import dgl
import torch

from dgl.dataloading import GraphDataLoader

from dgl.data import DGLDataset


class QM7bDataset(DGLDataset):
    _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat'
    _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(QM7bDataset, self).__init__(name='qm7b',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    # 将原始数据处理为图列表和标签列表。
    def process(self):
        mat_path = self.raw_path + '.mat'
        # 将数据处理为图列表和标签列表
        self.graphs, self.label = self._load_graph(mat_path)

    def __getitem__(self, idx):
        """ 通过idx获取对应的图和标签

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)

    @property
    def num_labels(self):
        """每个图的标签数，即预测任务数。"""
        return 14

# 数据导入
dataset = QM7bDataset()
num_labels = dataset.num_labels
print(num_labels)