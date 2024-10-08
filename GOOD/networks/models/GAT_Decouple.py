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
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing
from GOOD.utils.GATConv_Decouple import GATConv_Decouple
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
@register.model_register
class GAT_Decouple(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        in_channels = config.dataset.dim_node
        heads = config.model.num_heads
        out_channels = config.dataset.num_classes
        self.num_layers = config.model.model_layer
        dropout = config.model.dropout_rate
        alpha = config.train.alpha
        print('---', alpha)
        print('out', out_channels)
        hidden_channels = config.model.dim_hidden//heads
        self.model = config.model
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.props = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        # self.lin_src = Linear(hidden_channels*heads, heads * hidden_channels,
        #                       bias=False, weight_initializer='glorot')
        self.lin_src = Linear(hidden_channels*heads, heads * hidden_channels)
        # self.lin_dst = self.lin_src
        self.lin_dst = Linear(hidden_channels * heads, heads * hidden_channels)
        self.GAT_Prop = GATConv_Decouple(in_channels, hidden_channels, heads=heads, K=self.num_layers,alpha=alpha, concat=True)
        self.lin1 = Linear(in_channels, hidden_channels*heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        self.hi = Linear(hidden_channels * heads, hidden_channels*heads)
        self.bnhi = nn.BatchNorm1d(hidden_channels * heads)
        self.lin2 = Linear(hidden_channels*heads, config.dataset.num_classes)
        self.lin3 = Linear(config.model.dim_hidden*self.num_layers, hidden_channels)
        self.bn2 = nn.BatchNorm1d(config.model.dim_hidden)
        self.lr_att = nn.Linear(hidden_channels + hidden_channels, 1)
        self.act = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(0)
        self.att_drop = nn.Dropout(dropout)
        self.residual = False
        self.pre_dropout = True
        self.lr_output = Linear(hidden_channels*heads, out_channels)
        self.res_fc = nn.Linear(hidden_channels, hidden_channels)
        self.prelu = F.elu
        self.drop = dropout
        self.proj = Linear(hidden_channels, 1)
        for _ in range(self.num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for hop in range(self.num_layers):
            self.props.append(APPNP(hop+1, config.model.appnp_alpha))
        # self.dropout = dropout
        self.activation = F.elu
        self.act1 = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        for prop in self.props:
            prop.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.hi.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bnhi.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        out = self.bn1(self.lin1(x))
        out = F.relu(out)
        out = F.dropout(out, p=self.drop, training=self.training)
        # out = self.bnhi(self.hi(out))
        # out = F.relu(out)
        # out = F.dropout(out, p=self.drop, training=self.training)
        x_src = x_dst = self.lin_src(out)
        # x_dst = self.lin_dst(out)
        x = self.GAT_Prop(x_src, x_dst, edge_index)
        # x = 0.5*x+0.5*x_src
        # print('----', x.size())
        x = self.lin2(x)






    ###########################################################
        # for hop in range(self.num_layers):
        #     out = self.props[hop](out, edge_index)
        #     input_list.append(out)
        # num_node = input_list[0].shape[0]
        # concat_features = torch.cat(input_list, dim=1)
        # jk_ref = self.dropout(self.prelu(self.bn2(self.lin3(concat_features))))
        # # jk_ref = concat_features
        # attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
        #                     input_list]
        # W = torch.cat(attention_scores, dim=1)
        # # print('W--------------', attention_scores[0].shape)
        # W = F.softmax(W, 1)
        # right_1 = torch.mul(input_list[0], self.att_drop(
        #     W[:, 0].view(num_node, 1)))
        # for i in range(1, self.num_layers):
        #     right_1 = right_1 + \
        #         torch.mul(input_list[i], self.att_drop(
        #             W[:, i].view(num_node, 1)))

        ###################################33
        # pred = []
        # for i in range(self.num_layers):
        #     right_1 = torch.mul(input_list[i], self.att_drop(
        #         W[:, i].view(num_node, 1)))
        #     pred.append(right_1)
        # pps = torch.stack(pred, dim=1)
        # retain_score = self.proj(pps)
        # retain_score = retain_score.squeeze()
        # retain_score = torch.sigmoid(retain_score)
        # retain_score = retain_score.unsqueeze(1)
        # right_1 = torch.matmul(retain_score, pps).squeeze()
     #####################################33
        # if self.residual:
        #     right_1 += self.res_fc(input_list[0])
        #     right_1 = self.dropout(self.prelu(right_1))
        # if self.pre_dropout:
        #     right_1=self.dropout(self.prelu(right_1))
        # right_1 = self.lr_output(right_1)
        return x


















