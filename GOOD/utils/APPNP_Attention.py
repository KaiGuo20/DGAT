from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.nn import APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import is_torch_sparse_tensor, spmm, to_edge_index
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class APPNP_Attention(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        K : int = 1,
        alpha: float=0.0,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.alpha = alpha
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.prop1 = APPNP(self.K, alpha)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x_src, x_dst, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels
        x_orginal = x_src
        x_src = x_src.view(-1, H, C)
        x_dst = x_dst.view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # alpha_src = (x_src * self.att_src).sum(dim=-1)
        # alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        # alpha = (alpha_src, alpha_dst)

        edge_weight=None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
                edge_index1, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x_src.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x_src.dtype)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_weight = None
        # self.normalize = True
        # self._cached_edge_index = None
        # self._cached_adj_t = None
        # self.cached = False
        # if self.normalize:
        #     if isinstance(edge_index, Tensor):
        #         cache = self._cached_edge_index
        #         if cache is None:
        #             edge_index, edge_weight = gcn_norm(  # yapf: disable
        #                 edge_index, edge_weight, x_orginal.size(self.node_dim), False,
        #                 self.add_self_loops, self.flow, dtype=x_src.dtype)
        #             if self.cached:
        #                 self._cached_edge_index = (edge_index, edge_weight)
        #         else:
        #             edge_index, edge_weight = cache[0], cache[1]
        #
        #     elif isinstance(edge_index, SparseTensor):
        #         cache = self._cached_adj_t
        #         if cache is None:
        #             edge_index = gcn_norm(  # yapf: disable
        #                 edge_index, edge_weight, x_orginal.size(self.node_dim), False,
        #                 self.add_self_loops, self.flow, dtype=x_src.dtype)
        #             if self.cached:
        #                 self._cached_adj_t = edge_index
        #         else:
        #             edge_index = cache


        # alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)



        # for k in range(self.K):
        #     # propagate_type: (x: Tensor, edge_weight: OptTensor)
        #     x = self.propagate(edge_index, x=x, alpha=alpha, size=size)


        h = x_orginal

        for k in range(self.K):
            x_org = 0
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
            for i in range(self.heads):
                alpha1 = alpha[:, i] + edge_weight
                # edge_index1, alpha_norm = gcn_norm(  # yapf: disable
                #     edge_index, alpha[:,i], x_orginal.size(self.node_dim), False,
                #     self.add_self_loops, self.flow, dtype=x_src.dtype)
                # x_current = self.propagate(edge_index, x=x_orginal, edge_weight=alpha1, size=None)
                x_org = x_org+alpha1
            alpha1 = alpha1/self.K
            x_current = self.propagate(edge_index, x=x_orginal, edge_weight=alpha1, size=None)
            # x_src = x_dst = x_org
            # x_src = x_src.view(-1, H, C)
            # x_dst = x_dst.view(-1, H, C)
            #
            # x_orginal = x_org * (1 - self.alpha)
            # x_orginal = x_orginal + self.alpha * h



        # x = self.prop1(x, edge_index)
        out = x_current


        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    # def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
    #     return alpha.unsqueeze(-1) * x_j
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')