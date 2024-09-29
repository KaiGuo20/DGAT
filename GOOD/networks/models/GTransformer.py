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
from .Classifiers import Classifier
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool

import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


@register.model_register
class GTransformer(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        hops = config.model.hops
        print('hops----', hops)
        n_class = config.dataset.num_classes
        input_dim = config.dataset.dim_node
        pe_dim = config.model.pe_dim
        n_layers = config.model.model_layer
        num_heads = config.model.num_heads
        hidden_dim = config.model.dim_hidden
        ffn_dim = 64
        dropout_rate = config.model.dropout_rate
        attention_dropout_rate = config.model.attention_dropout_rate
        self.seq_len = hops + 1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads

        self.n_layers = n_layers
        self.n_class = n_class

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        self.model = config.model

        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        # self.bn = nn.BatchNorm1d(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.n_class)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    # def forward(self, batched_data):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        batched_data, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        tensor = self.att_embeddings_nope(batched_data)
        # tensor = batched_data
        # import ipdb;
        # ipdb.set_trace()
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        output = self.final_ln(tensor)

        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))

        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()
        # print('output', output.size())
        # output = self.bn(output)
        if self.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif self.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
        # output = self.readout(output, batch)
        output = self.Linear1(torch.relu(self.out_proj(output)))

        # return torch.log_softmax(output, dim=1)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]

        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x





