"""
GCN implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
<https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import numpy as np
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing, TransformerConv, GATConv
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
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
class SR_GPRGNN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
    <https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GPRGNN(config)
        # self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The SRGCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, features]

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = out_readout
        return out, out_readout



def KMM(X, Xtest, config: Union[CommonArgs, Munch], _A=None, _sigma=1e1, beta=0.2):
    r"""
    Kernel mean matching (KMM) to compute the weight for each training instance

    Args:
        X (Tensor): training instances to be matched
        Xtest (Tensor): IID samples to match the training instances
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)
        _A (numpy array): one hot matrix of the training instance labels
        _sigma (float): normalization term
        beta (float): regularization weight

    Returns:
        - KMM_weight (numpy array) - KMM_weight to match each training instance
        - MMD_dist (Tensor) - MMD distance

    """
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1), device=config.device))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A == 0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples, 1))

    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H.cpu().numpy().astype(np.double)), matrix(f.cpu().numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()


def pairwise_distances(x, y=None):
    r"""
    computation tool for pairwise distances

    Args:
        x (Tensor): a Nxd matrix
        y (Tensor): an optional Mxd matirx

    Returns (Tensor):
        dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def cmd(X, X_test, K=5):
    r"""
    central moment discrepancy (cmd). objective function for keras models (theano or tensorflow backend). Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.

    Args:
        X (Tensor): training instances
        X_test (Tensor): IID samples
        K (int): number of approximation degrees

    Returns (Tensor):
         central moment discrepancy

    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms)


def l2diff(x1, x2):
    r"""
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    r"""
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)