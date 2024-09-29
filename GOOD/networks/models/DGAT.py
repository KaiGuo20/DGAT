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
from GOOD.utils.APPNP_GATConv import APPNP_GATConv
from GOOD.utils.GATConv_alpha import GATConv_alpha
from GOOD.utils.GATConv2_alpha import GATConv2_alpha
from GOOD.utils.SGC_GATConv import SGC_GATConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

@register.model_register
class DGAT(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        num_features = config.dataset.dim_node
        hidden = config.model.dim_hidden
        heads = config.model.num_heads
        hidden_channels = hidden//config.model.num_heads
        num_classes = config.dataset.num_classes
        print('class', num_classes)
        num_layers = config.model.model_layer
        K = config.model.model_layer
        app_alpha = config.model.appnp_alpha
        self.dropout = config.model.dropout_rate
        self.attdrop = config.model.attention_dropout_rate
        self.att_alpha = config.train.alpha
        print('k=-----', K)
        self.lin1 = Linear(num_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        # self.prop1 = APPNP(K, alpha)
        self.prop1 = APPNP_GATConv(K, heads, app_alpha, self.att_alpha)
        self.lin3 = nn.Linear(hidden, config.dataset.num_classes)

        self.model = config.model
        self.dataset = config.dataset
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        M = config.model.AGAT_Layer
        if M==1:
            self.convs.append(
                    GATConv2_alpha(hidden, num_classes, heads=heads, concat=False, dropout=self.attdrop))
            self.bns.append(nn.BatchNorm1d(num_classes))
        if M==2:
            self.convs.append(
                GATConv2_alpha(hidden, hidden_channels, heads=heads, concat=True, dropout=self.attdrop))

            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
            for _ in range(2-1):
                self.convs.append(
                        GATConv2_alpha(hidden_channels*heads, num_classes, heads=heads, concat=False, dropout=self.attdrop))
                self.bns.append(nn.BatchNorm1d(num_classes))
        if M==3:
            self.convs.append(
                GATConv2_alpha(hidden, hidden_channels, heads=heads, concat=True, dropout=self.attdrop,
                              att_alpha=self.att_alpha))

            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.convs.append(
                GATConv2_alpha(hidden_channels*heads, hidden_channels, heads=heads, concat=True, dropout=self.attdrop,
                              att_alpha=self.att_alpha))

            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            for _ in range(2 - 1):
                self.convs.append(
                    GATConv2_alpha(hidden_channels * heads, num_classes, heads=heads, concat=False, dropout=self.attdrop,
                                  att_alpha=self.att_alpha))
                self.bns.append(nn.BatchNorm1d(num_classes))
        self.activation = F.elu
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        # import ipdb; ipdb.set_trace()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input = x
        x = self.lin2(x)
        for i, conv in enumerate(self.convs):
            input, alpha = conv(input, edge_index)
            input = self.bns[i](input)
            input = self.activation(input)
            input = F.dropout(input, p=self.dropout, training=self.training)
        x = self.prop1(alpha, x, edge_index)
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.readout(x, batch)
        # x = self.lin3(x)

        return x


