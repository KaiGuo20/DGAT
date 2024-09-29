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

import torch
import torch.nn as nn
from GOOD.utils.GCNII_layer import GCNIIdenseConv
from torch_geometric.nn import GCN2Conv
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
class GCNII(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()

        num_features = config.dataset.dim_node
        hidden = config.model.dim_hidden
        num_classes = config.dataset.num_classes
        K = config.model.model_layer
        self.alpha = config.train.alpha
        self.theta = config.model.theta
        self.dropout = config.model.dropout_rate
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, hidden))
        for layer in range(K):
           # self.convs.append(GCNIIdenseConv(hidden, hidden, bias='bn'))
            self.convs.append(GCN2Conv(hidden, self.alpha, self.theta, layer+1,
                         shared_weights=True, normalize=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.convs.append(torch.nn.Linear(hidden, num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())
        print('k=-----', K)
        self.reset_parameters()
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        _hidden = []
        x = F.relu(self.convs[0](x))
        x = F.dropout(x, self.dropout, training=self.training)
        _hidden.append(x)
        for i, con in enumerate(self.convs[1:-1]):
            x = self.bns[i](con(x, _hidden[0], edge_index))
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x


