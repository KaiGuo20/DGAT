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
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing, GATv2Conv, TransformerConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
# from GOOD.utils.GATConv import GATConv
@register.model_register
class GATv2(GNNBasic):
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
        hidden_channels = config.model.dim_hidden // heads
        self.attdrop = config.model.attention_dropout_rate
        self.att_alpha = config.train.alpha
        # print('hidden--', hidden_channels)
        self.model = config.model

        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=self.attdrop))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.lin2 = nn.Linear(hidden_channels*heads, config.dataset.num_classes)
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, dropout=self.attdrop))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATv2Conv(hidden_channels*heads, out_channels, heads=heads, concat=False, dropout=self.attdrop))

        self.dropout = dropout
        self.activation = F.elu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin2.reset_parameters()



    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.convs[-1](x, edge_index)
        x = self.readout(x, batch)


        # print(x.size())
        return x

