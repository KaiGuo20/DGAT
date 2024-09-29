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
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
import argparse
import os.path as osp
from typing import Any, Dict, Optional
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, GINEConv, GINConv
from .PerformerAttention import PerformerAttention
from .GPSConv import GPSConv

@register.model_register
class GPS(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(self, channels: int, pe_dim: int, num_layers: int,
    #              attn_type: str, attn_kwargs: Dict[str, Any]):
    #     super().__init__()
        channels = config.model.dim_hidden
        num_layers = config.model.model_layer
        self.input_dim = 9
        self.n_class = config.dataset.num_classes
        attn_kwargs = {'dropout': 0.5}
        attn_type = "multihead"
        # self.node_emb = Embedding(28, channels - pe_dim)
        # self.pe_lin = Linear(20, pe_dim)
        # self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)
        self.att_embeddings_nope = torch.nn.Linear(self.input_dim, channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            # nn = Sequential(
            #     Linear(channels, channels),
            #     ReLU(),
            #     Linear(channels, self.n_class),
            # )
            out_channels = self.n_class
            # print('node', self.input_dim)

            conv = GPSConv(channels, GCNConv(channels, channels), heads=4)
            self.convs.append(conv)
        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            # Linear(channels // 2, channels // 4),
            # ReLU(),
            Linear(channels // 2, self.n_class),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # x, edge_index, edge_weight, batch = self.arguments_read(*args, **
        x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
        # x_pe = self.pe_norm(pe)
        # x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        x = x.float()
        # print('x',x.type())
        x = self.att_embeddings_nope(x)
        # import ipdb; ipdb.set_trace()
        edge_attr = None
        # print('edge', edge_index.size())
        # print('xx', x.size(0))
        # print('batch', batch)
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        # x = global_add_pool(x, batch)
        # if self.model.model_level == 'node':
        #     self.readout = IdenticalPool()
        # elif self.model.global_pool == 'mean':
        #     self.readout = GlobalMeanPool()
        # else:
        #     self.readout = GlobalMaxPool()
        self.readout = GlobalMeanPool()
        x = self.readout(x, batch, batch_size)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


