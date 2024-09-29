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
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from GOOD.utils.GATConv import GATConv
from torch_geometric.nn.dense.linear import Linear

from GOOD.utils.SimpleConv import SimpleConv


@register.model_register
class PMLP(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        in_channels = config.dataset.dim_node

        out_channels = config.dataset.num_classes
        num_layers = config.model.model_layer
        dropout = config.model.dropout_rate
        heads = config.model.num_heads
        hidden_channels = config.model.dim_hidden
        # print('hidden--', hidden_channels)
        self.model = config.model

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = True
        norm = True

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, self.bias))
        for _ in range(self.num_layers - 2):
            lin = Linear(hidden_channels, hidden_channels, self.bias)
            self.lins.append(lin)
        self.lins.append(Linear(hidden_channels, out_channels, self.bias))

        self.norm = None
        if norm:
            self.norm = torch.nn.BatchNorm1d(
                hidden_channels,
                affine=False,
                track_running_stats=False,
            )
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight, gain=1.414)
            if self.bias:
                torch.nn.init.zeros_(lin.bias)


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        if not self.training and edge_index is None:
            raise ValueError(f"'edge_index' needs to be present during "
                             f"inference in '{self.__class__.__name__}'")

        for i in range(self.num_layers):
            x = x @ self.lins[i].weight.t()
            if not self.training:
                x = self.conv(x, edge_index)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

