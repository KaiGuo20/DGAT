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

    def forward(self, alpha: Tensor, x: Tensor, edge_index: Adj,
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
        for k in range(self.K):
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

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            # for i in range(self.heads):
            #     alpha1 = alpha[:, i] + edge_weight
            #     # edge_index1, alpha_norm = gcn_norm(  # yapf: disable
            #     #     edge_index, alpha[:,i], x_orginal.size(self.node_dim), False,
            #     #     self.add_self_loops, self.flow, dtype=x_src.dtype)
            #     # x_current = self.propagate(edge_index, x=x_orginal, edge_weight=alpha1, size=None)
            #     x_org = x_org+alpha1
            # alpha1 = alpha1/self.K

            # for i in range(self.heads):
            #     # edge_index1, alpha[:,i] = gcn_norm(  # yapf: disable
            #     #     edge_index, alpha[:,i], x_src.size(self.node_dim), False,
            #     #     self.add_self_loops, self.flow, dtype=x_src.dtype)
            #     # alpha[:, i] = (1 - self.att_alpha) * alpha[:, i] + self.att_alpha * edge_weight
            #     edge_weight = (1 - self.att_alpha) * alpha[:, i] + self.att_alpha * edge_weight
            edge_weight = (1 - self.att_alpha) * alpha1 + self.att_alpha * edge_weight
            # print('edge_weight', edge_weight.size())
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.app_alpha)
            x = x + self.app_alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'