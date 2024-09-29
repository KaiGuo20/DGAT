r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size
from torch_sparse import SparseTensor

from GOOD import register
from torch_geometric.utils import to_scipy_sparse_matrix
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from torch_geometric.utils import to_dense_adj
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

@register.model_register
class GAT_Pro(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        num_features = config.dataset.dim_node
        hidden = config.model.dim_hidden
        heads = config.model.num_heads
        hidden_channels = config.model.dim_hidden // heads
        num_classes = config.dataset.num_classes
        K = config.model.model_layer
        alpha = config.model.appnp_alpha
        self.dropout = config.model.dropout_rate
        print('k=-----', K)
        self.lin1 = GATConv(num_features, hidden_channels, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        self.lin2 = GATConv(hidden_channels*heads, num_classes, heads=heads, concat=False)
        self.prop1 = APPNP(K, alpha)
        self.model = config.model
        self.dataset = config.dataset
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        # from torch_geometric.utils import degree
        # node_index = torch.arange(x.size(0))
        # x = x/degree(edge_index[0])[node_index].view(-1,1)
        if self.dataset.dataset_name == "GOODCBAS":
            x = self.prop1(x, edge_index)
            print('CBAS----')
            x = self.bn1(self.lin1(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.model.model_level == 'node':
                self.readout = IdenticalPool()
            elif self.model.global_pool == 'mean':
                self.readout = GlobalMeanPool()
            else:
                self.readout = GlobalMaxPool()
            x = self.readout(x, batch)
            x = self.lin2(x)
        else:

            # print(edge_index.shape)
            # torch.set_printoptions(profile="full")
            # N = edge_index.max().item() + 1
            # # 创建一个空的 N x N 邻接矩阵
            # adj_matrix = torch.zeros(N, N)
            # # 使用 edge_index 填充邻接矩阵
            # adj_matrix[edge_index[0], edge_index[1]] = 1
            #
            # print(adj_matrix.size())
            # a2 = torch.mm(adj_matrix, adj_matrix)
            # a3 = torch.mm(a2, adj_matrix)
            # a4 = torch.mm(a3, adj_matrix)
            # print(a4)
            x = self.bn1(self.lin1(x, edge_index))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x, edge_index)
            x = self.prop1(x, edge_index)
            if self.model.model_level == 'node':
                self.readout = IdenticalPool()
            elif self.model.global_pool == 'mean':
                self.readout = GlobalMeanPool()
            else:
                self.readout = GlobalMaxPool()
            x = self.readout(x, batch)
        return x


