from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing

@register.model_register
class Mixup_GPRGNN(GNNBasic):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    # def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', dropout=.5,
    #         lr=0.01, weight_decay=0, device='cpu',
    #         K=10, alpha=.1, Gamma=None, ppnp='GPR_prop', args=None):
    #     super(GPRGNN, self).__init__()

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)

        in_channels = config.dataset.dim_node
        hidden_channels = config.model.dim_hidden
        out_channels = config.dataset.num_classes
        Init = 'PPR'
        ppnp = 'GPR_prop'
        self.K = 10
        self.alpha = config.model.appnp_alpha
        Gamma = None

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        # self.args = args

        if ppnp == 'PPNP':
            self.prop1 = APPNP(self.K, self.alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(self.K, self.alpha, Init, Gamma)

        self.Init = Init
        self.dprate = 0.0
        self.dropout = config.model.dropout_rate
        self.name = "GPR"
        # self.weight_decay = weight_decay
        # self.lr = lr
        # self.device=device
        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(self.K + 1)
            TEMP[self.alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = self.alpha * (1 - self.alpha) ** np.arange(self.K + 1)
            TEMP[-1] = (1 - self.alpha) ** self.K

        self.temp = nn.Parameter(torch.tensor(TEMP))
        self.model = config.model

        self.reset_parameters()

    # def initialize(self):
    #     self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.prop1.reset_parameters()
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K
        self.lin_out.reset_parameters()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ood_algorithm = kwargs.get('ood_algorithm')
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        h_a = [x]

        if edge_weight is not None:
            edge_index = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
            if self.dprate == 0.0:
                for k in range(self.K):
                    x = self.prop1(h_a[-1],  self.temp, h_a[-1], edge_index, k)
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
                        new_h_a = self.prop1(h_a[0],self.temp, h_mix[0], edge_index_a, k=i)
                        new_h_b = self.prop1(h_b[0],self.temp, h_mix[0], edge_index_b, k=i)
                    else:
                        new_h_a = self.prop1(h_a[-1], self.temp, h_mix[-1], edge_index=edge_index_a, k=i)
                        new_h_b = self.prop1(h_b[-1],self.temp, h_mix[-1], edge_index=edge_index_b, k=i)
                h_mix.append(F.dropout((lam * new_h_a + (1 - lam) * new_h_b), p=self.dropout, training=self.training))
                x = h_mix[-1]


                if self.model.model_level == 'node':
                    self.readout = IdenticalPool()
                elif self.model.global_pool == 'mean':
                    self.readout = GlobalMeanPool()
                else:
                    self.readout = GlobalMaxPool()
                # x = self.lin_out(x)

                x = self.readout(x, batch)
        return x

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha


    def forward(self, x, temp, x_cen, edge_index, k, edge_weight=None):

        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(temp[0])

        x = self.propagate(edge_index, x=x, norm=norm)
        gamma = temp[k+1]
        hidden = hidden + gamma*x + x_cen
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


