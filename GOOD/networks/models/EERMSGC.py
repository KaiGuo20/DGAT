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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, APPNP, MessagePassing, TransformerConv, GATConv, SGConv
from torch.nn import Linear
class SGC_origin(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
    #              dropout=0.5, heads=2):
    #     super(GAT, self).__init__()
        num_features = config.dataset.dim_node
        hidden = config.model.dim_hidden
        num_classes = config.dataset.num_classes
        K = config.model.model_layer
        alpha = config.model.appnp_alpha
        self.dropout = config.model.dropout_rate
        print('k=-----', K)
        self.conv1 = SGConv(num_features, num_classes, K=K,
                            cached=False)
        self.model = config.model
        self.dataset = config.dataset
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.lin1.reset_parameters()
    #     self.lin2.reset_parameters()
    #     self.bn1.reset_parameters()
        # self.lin3.reset_parameters()
        # self.bn2.reset_parameters()
        # self.lin4.reset_parameters()
        # self.bn3.reset_parameters()
        # self.lin5.reset_parameters()
        # self.bn4.reset_parameters()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        # from torch_geometric.utils import degree
        # node_index = torch.arange(x.size(0))
        # x = x/degree(edge_index[0])[node_index].view(-1,1)
        if self.dataset.dataset_name == "GOODCBAS":
            x = self.prop1(x, edge_index)
            print('CBAS----')
            x = self.bn1(self.lin1(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.model.model_level == 'node':
                self.readout = IdenticalPool()
            elif self.model.global_pool == 'mean':
                self.readout = GlobalMeanPool()
            else:
                self.readout = GlobalMaxPool()
            x = self.readout(x, batch)
            x = self.lin2(x)
        else:
            x = self.conv1(x, edge_index)
            if self.model.model_level == 'node':
                self.readout = IdenticalPool()
            elif self.model.global_pool == 'mean':
                self.readout = GlobalMeanPool()
            else:
                self.readout = GlobalMaxPool()
            x = self.readout(x, batch)
        return x
@register.model_register
class EERMSGC(GNNBasic):
    r"""
    EERM implementation adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, config):
        super(EERMSGC, self).__init__(config)
        # self.gnn = GCNFeatExtractor(config)#config.model.model_name(config)
        # config.model.model_name = GCN2(config)
        self.gnn = SGC_origin(config)
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
