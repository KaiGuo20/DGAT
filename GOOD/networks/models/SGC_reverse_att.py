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
from GOOD.utils.GATConv_alpha import GATConv_alpha
from GOOD.utils.SGC_GATConv import SGC_GATConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from GOOD.utils.GATConv import GATConv
from GOOD.utils.SGConv import SGConv
@register.model_register
class SGC_reverse_att(GNNBasic):
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
        hidden = hidden_channels // config.model.num_heads
        self.attdrop = config.model.attention_dropout_rate
        self.att_alpha = config.train.alpha
        # print('hidden--', hidden_channels)
        self.model = config.model
        self.SGC = SGC_GATConv(heads, in_channels, hidden_channels, K=num_layers,
                            cached=False)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins.append(
            nn.Linear(in_channels, hidden_channels))

        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lin_out = nn.Linear(hidden_channels, config.dataset.num_classes)
        for _ in range(num_layers-1):

            self.lins.append(
                    nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bn_atts = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv_alpha(in_channels, hidden, heads=heads, concat=True, dropout=self.attdrop, att_alpha=self.att_alpha))

        self.bn_atts.append(nn.BatchNorm1d(hidden*heads))
        for _ in range(2-1):
            self.convs.append(
                    GATConv_alpha(hidden*heads, hidden, heads=heads, concat=False, dropout=self.attdrop, att_alpha=self.att_alpha))
            self.bn_atts.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.activation = F.relu
        self.act = F.elu

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn_att in self.bn_atts:
            bn_att.reset_parameters()
        self.lin_out.reset_parameters()


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        input = x
        for i, lin in enumerate(self.lins):
            x = lin(x)
            x = self.bns[i](x)
            if i < len(self.lins) - 1:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            input, alpha = conv(input, edge_index)
            input = self.bn_atts[i](input)
            input = self.act(input)
            input = F.dropout(input, p=self.dropout, training=self.training)

        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.SGC(alpha, x, edge_index)
        x = self.lin_out(x)
        x = self.readout(x, batch)



        # print(x.size())
        return x

