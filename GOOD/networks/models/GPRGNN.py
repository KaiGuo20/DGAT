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
class GPRGNN(GNNBasic):
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
        K = 10
        alpha = config.model.appnp_alpha
        Gamma = None

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        # self.args = args

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = 0.0
        self.dropout = config.model.dropout_rate
        self.name = "GPR"
        # self.weight_decay = weight_decay
        # self.lr = lr
        # self.device=device

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
            if self.dprate == 0.0:
                x = self.prop1(x, adj)
            else:
                x = F.dropout(x, p=self.dprate, training=self.training)
                x = self.prop1(x, adj)
        else:
            if self.dprate == 0.0:

                x = self.prop1(x, edge_index, edge_weight)
            else:
                x = F.dropout(x, p=self.dprate, training=self.training)
                x = self.prop1(x, edge_index, edge_weight)

        return x
        # return F.log_softmax(x, dim=1)

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        # print('edge----', edge_index.size(-1))
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


