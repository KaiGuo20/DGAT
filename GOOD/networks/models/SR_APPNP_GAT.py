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
from GOOD.utils.APPNP_GATConv import APPNP_GATConv
from GOOD.utils.GATConv_alpha import GATConv_alpha

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing, TransformerConv, GATConv
from torch.nn import Linear
class APPNP_GAT(GNNBasic):
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
        app_alpha = config.model.appnp_alpha
        self.dropout = config.model.dropout_rate
        self.attdrop = config.model.attention_dropout_rate
        self.att_alpha = config.train.alpha
        print('k=-----', K)
        self.lin1 = Linear(num_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        # self.prop1 = APPNP(K, alpha)
        self.prop1 = APPNP_GATConv(K, heads, app_alpha, self.att_alpha)

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

        # import ipdb; ipdb.set_trace()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        x = self.bn1(self.lin1(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input = x
        x = self.lin2(x)
        for i, conv in enumerate(self.convs):
            input, alpha = conv(input, edge_index)
            input = self.bns[i](input)
            input = self.activation(input)
            input = F.dropout(input, p=self.dropout, training=self.training)
        x = self.prop1(alpha, x, edge_index)
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        x = self.readout(x, batch)

        return x


@register.model_register
class SR_APPNP_GAT(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
    <https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = APPNP_GAT(config)
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