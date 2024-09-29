r"""
The implementation of `Handling Distribution Shifts on Graphs: An Invariance Perspective <https://arxiv.org/abs/2202.02466>`_.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph

from GOOD import register
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
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
class EERMAPPNP_GAT(GNNBasic):
    r"""
    EERM implementation adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, config):
        super(EERMAPPNP_GAT, self).__init__(config)
        # self.gnn = GCNFeatExtractor(config)#config.model.model_name(config)
        # config.model.model_name = GCN2(config)
        self.gnn = APPNP_GAT(config)
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
