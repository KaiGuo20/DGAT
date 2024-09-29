r"""
The implementation of `Handling Distribution Shifts on Graphs: An Invariance Perspective <https://arxiv.org/abs/2202.02466>`_.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from GOOD import register
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing, TransformerConv, GATConv
from torch.nn import Linear
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

@register.model_register
class EERMGPRGNN(GNNBasic):
    r"""
    EERM implementation adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, config):
        super(EERMGPRGNN, self).__init__(config)
        # self.gnn = GCNFeatExtractor(config)#config.model.model_name(config)
        # config.model.model_name = GCN2(config)
        self.gnn = GPRGNN(config)
        self.p = 0.2
        self.K = config.ood.extra_param[0]
        self.T = config.ood.extra_param[1]
        self.num_sample = config.ood.extra_param[2]
        self.classifier = Classifier(config)

        self.gl = Graph_Editer(self.K, config.dataset.num_train_nodes, config.device)
        self.gl.reset_parameters()
        self.gl_optimizer = torch.optim.Adam(self.gl.parameters(), lr=config.ood.extra_param[3])

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')
        loss_func = self.config.metric.loss_func

        # --- K fold ---
        if self.training:
            edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
            x = data.x[data.train_mask]
            y = data.y[data.train_mask]

            # --- check will orig_edge_index change? ---
            orig_edge_index = edge_index
            for t in range(self.T):
                Loss, Log_p = [], 0
                for k in range(self.K):
                    edge_index, log_p = self.gl(orig_edge_index, self.num_sample, k)
                    raw_pred = self.gnn(data=Data(x=x, edge_index=edge_index, y=y))

                    loss = loss_func(raw_pred, y)
                    Loss.append(loss.view(-1))
                    Log_p += log_p
                Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
                reward = Var.detach()
                inner_loss = - reward * Log_p
                self.gl_optimizer.zero_grad()
                inner_loss.backward()
                self.gl_optimizer.step()
            return Var, Mean
        else:
            out = self.gnn(data=data)
            return out


class Graph_Editer(nn.Module):
    r"""
    EERM's graph editer adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.n = n
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, num_sample, k):
        n = self.n
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p
