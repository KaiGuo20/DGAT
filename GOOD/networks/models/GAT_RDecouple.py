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
from torch.nn import Linear
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch_geometric.nn import SGConv, APPNP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
@register.model_register
class GAT_RDecouple(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        in_channels = config.dataset.dim_node

        out_channels = config.dataset.num_classes
        self.num_layers = config.model.model_layer
        dropout = config.model.dropout_rate
        heads = config.model.num_heads
        hidden_channels = config.model.dim_hidden
        #print('hidden--', hidden_channels)
        self.model = config.model

        self.convs = nn.ModuleList()

        self.bns = nn.ModuleList()
        self.props = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.lin1 = Linear(in_channels, config.model.dim_hidden)
        self.bn1 = nn.BatchNorm1d(config.model.dim_hidden)
        self.lin2 = Linear(config.model.dim_hidden, hidden_channels)

        self.lr_att = nn.Linear(hidden_channels + hidden_channels, 1)
        self.act = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(0)
        self.att_drop = nn.Dropout(dropout)
        self.residual = False
        self.pre_dropout = False
        self.lr_output = Linear(hidden_channels, out_channels)
        self.res_fc = nn.Linear(hidden_channels, hidden_channels)
        self.prelu = F.elu
        self.drop = dropout
        for _ in range(self.num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for hop in range(self.num_layers + 1):
            self.props.append(APPNP(3, config.model.appnp_alpha))
        # self.dropout = dropout
        self.activation = F.elu

    def reset_parameters(self):
        for prop in self.props:
            prop.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        input_list = []
        for hop in range(self.num_layers):
            out = self.bns[hop](self.lin1(x))
            out = F.relu(out)
            out = F.dropout(out, p=self.drop, training=self.training)
            out = self.lin2(out)
            out = self.props[hop](out, edge_index)
            input_list.append(out)
        num_node = input_list[0].shape[0]
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], input_list[0]], dim=1))))
        for i in range(1, self.num_layers):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                    torch.mul(input_list[j], self.att_drop(
                        att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        for i in range(1, self.num_layers):
            right_1 = 0 + \
                torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(input_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        if self.pre_dropout:
            right_1=self.dropout(right_1)
        right_1 = self.lr_output(right_1)
        return right_1


















