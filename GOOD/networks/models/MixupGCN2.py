"""
GCN implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


@register.model_register
class Mixup_GCN2(GNNBasic):
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

        self.convs = nn.ModuleList()
        self.convs.append(
            MixUpGCNConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lin2 = nn.Linear(hidden_channels, config.dataset.num_classes)
        for _ in range(num_layers - 2):

            self.convs.append(
                    MixUpGCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        ood_algorithm = kwargs.get('ood_algorithm')
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        h_a = [x]

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(h_a[-1],h_a[-1], edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h_a.append(x)

        h_b = []
        for h in h_a:
            h_b.append(h[ood_algorithm.id_a2b])

        edge_index_a, edge_weight_a = edge_index, edge_weight
        if self.training:
            edge_index_b, edge_weight_b = ood_algorithm.data_perm.edge_index, edge_weight
        else:
            edge_index_b, edge_weight_b = edge_index, edge_weight

        lam = ood_algorithm.lam
        h_mix = [lam * h_a[0] + (1 - lam) * h_b[0]]

        for i, conv in enumerate(self.convs[:-1]):
            if i == 0:
                new_h_a = F.dropout(self.activation(self.bns[i](conv(h_a[0], h_mix[0], edge_index_a, edge_weight_a))),
                                    p=self.dropout, training=self.training)
                new_h_b = F.dropout(self.activation(self.bns[i](conv(h_b[0], h_mix[0], edge_index_b, edge_weight_b))),
                                    p=self.dropout, training=self.training)
            else:
                new_h_a = F.dropout(self.activation(self.bns[i](conv(h_a[-1], h_mix[-1], edge_index_a, edge_weight_a))), p=self.dropout,training=self.training)
                new_h_b = F.dropout(self.activation(self.bns[i](conv(h_b[-1], h_mix[-1], edge_index_b, edge_weight_b))), p=self.dropout,training=self.training)
            h_mix.append(F.dropout((lam * new_h_a + (1 - lam) * new_h_b), p=self.dropout, training=self.training))

        h_out = h_mix[-1]
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.readout(h_out, batch)
        x = self.convs[-1](x, edge_index)
        return x

class MixUpGCNConv(gnn.MessagePassing):
    r"""The graph convolutional operator from the `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.lin_cen = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_cen: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None) + self.lin_cen(x_cen)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
