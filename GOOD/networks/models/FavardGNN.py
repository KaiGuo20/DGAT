r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
"""
import torch
import torch.nn as nn
import torch as th
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
@register.model_register
class FavardGNN(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        in_feats = config.dataset.dim_node

        n_classes = config.dataset.num_classes
        num_layers = config.model.model_layer
        dropout = config.model.dropout_rate
        heads = config.model.num_heads
        n_hidden = config.model.dim_hidden
        # print('hidden--', hidden_channels)
        self.model = config.model

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(FavardNormalConv())
        self.K = num_layers
        self.n_channel = n_hidden
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.act_fn = F.relu

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.init_alphas()
        self.init_betas_and_yitas()

    def init_alphas(self):
        t = th.zeros(self.K + 1)
        t[0] = 1
        t = t.repeat(self.n_channel, 1)
        self.alpha_params = nn.Parameter(t.float())

    def init_betas_and_yitas(self):
        self.yitas = nn.Parameter(th.zeros(self.K + 1).repeat(self.n_channel, 1).float())  # (n_channels, K+1)
        self.sqrt_betas = nn.Parameter(th.ones(self.K + 1).repeat(self.n_channel, 1).float())  # (n_channels, K+1)
        return


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        x = self.dropout(x)
        x = self.fcs[0](x)

        x = self.act_fn(x)
        x = self.dropout2(x)

        sqrt_betas = th.clamp(self.sqrt_betas, 1e-2)
        # sqrt_betas = th.clamp(self.sqrt_betas, 1e-1)

        h0 = x / sqrt_betas[:, 0]
        rst = th.zeros_like(h0)
        rst = rst + self.alpha_params[:, 0] * h0

        last_h = h0
        second_last_h = th.zeros_like(h0)
        for i, con in enumerate(self.convs, 1):
            h_i = con(edge_index, edge_weight, x, last_h, second_last_h, self.yitas[:, i - 1], sqrt_betas[:, i - 1],
                      sqrt_betas[:, i])
            rst = rst + self.alpha_params[:, i] * h_i
            second_last_h = last_h
            last_h = h_i

        rst = self.dropout(rst)
        rst = self.fcs[-1](rst)
        return rst

class FavardNormalConv(MessagePassing):
    def __init__(self, fixed=False, kwargs={}):
        super(FavardNormalConv, self).__init__()

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, edge_weight,x, last_h, second_last_h, yita, sqrt_beta, _sqrt_beta):
        '''
        last_h:         N x C
        second_last_h : N x C
        yita:           C
        sqrt_beta:      C
        _sqrt_beta:     C
        '''
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm_A = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        rst = rst - yita.unsqueeze(0)*last_h - sqrt_beta.unsqueeze(0)*second_last_h
        rst = rst / _sqrt_beta.unsqueeze(0)
        return rst