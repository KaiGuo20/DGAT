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
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing
# from torch_geometric.nn.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
@register.model_register
class MLP(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        in_channels = config.dataset.dim_node

        out_channels = config.dataset.num_classes
        num_layers = config.model.model_layer
        dropout = config.model.dropout_rate
        hidden_channels = config.model.dim_hidden
        # print('hidden--', hidden_channels)
        self.model = config.model

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):

            self.convs.append(
                    nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x1 = self.readout(x, batch)
        x = self.convs[-1](x1)
        return x


# class MLP_Layer(MessagePassing):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         improved: bool = False,
#         cached: bool = False,
#         add_self_loops: bool = True,
#         normalize: bool = True,
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.improved = improved
#         self.cached = cached
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize
#
#         self._cached_edge_index = None
#         self._cached_adj_t = None
#
#         self.lin = nn.Linear(in_channels, out_channels, bias=False,
#                           weight_initializer='glorot')
#
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
# def reset_parameters(self):
#     super().reset_parameters()
#     self.lin.reset_parameters()
#     self._cached_edge_index = None
#     self._cached_adj_t = None
#
# def forward(self, x: Tensor, edge_index: Adj,
#                 edge_weight: OptTensor = None) -> Tensor:
#
#     if isinstance(x, (tuple, list)):
#         raise ValueError(f"'{self.__class__.__name__}' received a tuple "
#                          f"of node features as input while this layer "
#                          f"does not support bipartite message passing. "
#                          f"Please try other layers such as 'SAGEConv' or "
#                          f"'GraphConv' instead")
#
#     out = self.lin(x)
#
#     # propagate_type: (x: Tensor, edge_weight: OptTensor)
#
#     if self.bias is not None:
#         out = out + self.bias
#
#     return out
#
# def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
#
# def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#     return spmm(adj_t, x, reduce=self.aggr)
#
