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
from GOOD.utils.GATConv_alpha import GATConv_alpha
from GOOD.utils.SGC_GATConv import SGC_GATConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index
from torch_geometric.utils.sparse import set_sparse_value

@register.model_register
class Mixup_APPNP_GAT1(GNNBasic):
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
        self.K = K

        self.dropout = config.model.dropout_rate
        self.attdrop = config.model.attention_dropout_rate
        self.att_alpha = config.train.alpha
        app_alpha = config.model.appnp_alpha
        print('k=-----', K)
        self.lin1 = Linear(num_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lin2 = Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(num_classes)
        self.lin_out = Linear(hidden, num_classes)
        self.gcn = GCNConv(hidden, num_classes)
        # self.prop1 = APPNP(K, alpha)
        self.prop1 = APPNP_GATConv(K, heads, app_alpha, self.att_alpha)
        # self.prop1 = SGC_GATConv(heads, num_features, hidden_channels, K=num_layers,
        #                     cached=False)
        self.model = config.model
        self.dataset = config.dataset
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()

        M = config.model.AGAT_Layer
        if M==1:
            self.convs.append(
                    GATConv_alpha(hidden, num_classes, heads=heads, concat=False, dropout=self.attdrop))
            self.bns.append(nn.BatchNorm1d(num_classes))
        if M==2:
            self.convs.append(
                GATConv_alpha(hidden, hidden_channels, heads=heads, concat=True, dropout=self.attdrop))

            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
            for _ in range(2-1):
                self.convs.append(
                        GATConv_alpha(hidden_channels*heads, num_classes, heads=heads, concat=False, dropout=self.attdrop))
                self.bns.append(nn.BatchNorm1d(num_classes))
        if M==3:
            self.convs.append(
                GATConv_alpha(hidden, hidden_channels, heads=heads, concat=True, dropout=self.attdrop,
                              att_alpha=self.att_alpha))

            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.convs.append(
                GATConv_alpha(hidden_channels*heads, hidden_channels, heads=heads, concat=True, dropout=self.attdrop,
                              att_alpha=self.att_alpha))

            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            for _ in range(2 - 1):
                self.convs.append(
                    GATConv_alpha(hidden_channels * heads, num_classes, heads=heads, concat=False, dropout=self.attdrop,
                                  att_alpha=self.att_alpha))
                self.bns.append(nn.BatchNorm1d(num_classes))
        self.activation = F.elu
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.gcn.reset_parameters()

        # import ipdb; ipdb.set_trace()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ood_algorithm = kwargs.get('ood_algorithm')
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)

        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input = x
        x = self.lin2(x)
        h_a = [x]
        for i, conv in enumerate(self.convs):
            input, alpha = conv(input, edge_index)
            input = self.bns[i](input)
            input = self.activation(input)
            input = F.dropout(input, p=self.dropout, training=self.training)
        # x = self.prop1(alpha, x, edge_index, ood_algorithm1, edge_index_b, lam)
        for i in range(self.K):
            x = self.prop1(alpha, h_a[-1], h_a[-1], edge_index)
            h_a.append(F.dropout(x, p=self.dropout, training=self.training))
        h_b = []
        for h in h_a:
            h_b.append(h[ood_algorithm.id_a2b])

        edge_index_a, edge_weight_a = edge_index, edge_weight
        if self.training:
            edge_index_b, edge_weight_b = ood_algorithm.data_perm.edge_index, edge_weight
        else:
            edge_index_b, edge_weight_b = edge_index, edge_weight
        # --- Begin mixup: a mix b

        lam = ood_algorithm.lam
        h_mix = [lam * h_a[0] + (1 - lam) * h_b[0]]
        for i in range(self.K):
            if i ==0:
                new_h_a = self.prop1(alpha, h_a[0], h_mix[0], edge_index_a, edge_weight_a)
                new_h_b = self.prop1(alpha, h_b[0], h_mix[0], edge_index_b, edge_weight_b)
            else:
                new_h_a = self.prop1(alpha, h_a[-1], h_mix[-1], edge_index_a, edge_weight_a)
                new_h_b = self.prop1(alpha, h_b[-1], h_mix[-1], edge_index_b, edge_weight_b)
        h_mix.append(F.dropout((lam * new_h_a + (1 - lam) * new_h_b), p=self.dropout, training=self.training))
        x = h_mix[-1]




        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.lin_out(x)

        x = self.readout(x, batch)

        return x


class APPNP_GATConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, heads: int, app_alpha: float, att_alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.app_alpha = app_alpha
        self.heads = heads
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.att_alpha = att_alpha
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None


    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, alpha: Tensor, x: Tensor, x_cen: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        alpha1 = alpha.mean(dim=-1)

        if self.dropout > 0 and self.training:
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    _, edge_weight = to_edge_index(edge_index)
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                    edge_index = set_sparse_value(edge_index, edge_weight)
                else:
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
            else:
                value = edge_index.storage.value()
                assert value is not None
                value = F.dropout(value, p=self.dropout)
                edge_index = edge_index.set_value(value, layout='coo')

        edge_weight = (1 - self.att_alpha) * alpha1 + self.att_alpha * edge_weight
        # print('edge_weight', edge_weight.size())
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                           size=None)
        x = x * (1 - self.app_alpha)
        x = x + self.app_alpha * h
        x = x + x_cen

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'


