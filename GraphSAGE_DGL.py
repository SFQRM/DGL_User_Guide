import argparse
import torch as th
from dgl.data import RedditDataset
import dgl
import numpy as np


def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

"""
#### Entry point
def run(args, device, data):
    # Unpack data, in_feats=602 ,nodes=232965 ,edges=114848857,n_classes=41，train_nid 13w训练样本的id
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0]) #np.nonzeros()返回元组(分别描述⾮0元素的位置⼆维)
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    # Create sampler初始化，默认的fanout是10,25，这个的意思是⼀阶抽10倍，2阶抽25倍
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])
    # Create PyTorch DataLoader for constructing blocks,train—id是15w的数据索引，batch=1000，sampler抽样器，
    dataloader = DataLoader(
    dataset=train_nid.numpy(),
    batch_size=args.batch_size,
    collate_fn=sampler.sample_blocks, #样本不能被batch整除时，需要的处理函数，这⾥其实是对1000个种⼦id做抽样，返回block⼆部图的⽅法
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers)
    # Define model and optimizer ，输⼊维度602，隐层16,n_classes =41
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
    tic = time.time()
    # Loop over the dataloader to sample the computation dependency graph as a list of
    # blocks.
    for step, blocks in enumerate(dataloader):
"""

if __name__ == '__main__':
    """
    # 参数配置
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')
    """

    data = RedditDataset(self_loop=True)
    print(data)
    train_mask = data.train_mask
    # train_mask = data.ndata['train_mask']
    print(train_mask)
    val_mask = data.val_mask
    print(val_mask)
    features = th.Tensor(data.features)
    print(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels

    # Construct graph
    # g = dgl.graph(data.graph.all_edges())
    # g.ndata['features'] = features
    # prepare_mp(g)
    # Pack data
    # data = train_mask, val_mask, in_feats, labels, n_classes, g

    # run(args, device, data)
