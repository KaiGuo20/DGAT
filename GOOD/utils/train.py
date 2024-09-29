r"""Training utils.
"""
from typing import Union

import torch
from munch import Munch
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add
from GOOD.utils.args import CommonArgs


def nan2zero_get_mask(data, task, config: Union[CommonArgs, Munch]):
    r"""
    Training data filter masks to process NAN.

    Args:
        data (Batch): input data
        task (str): mask function type
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.model_level`)

    Returns (Tensor):
        [mask (Tensor) - NAN masks for data formats, targets (Tensor) - input labels]

    """
    if config.model.model_level == 'node':
        if 'train' in task:
            mask = data.train_mask
        elif task == 'id_val':
            mask = data.get('id_val_mask')
        elif task == 'id_test':
            mask = data.get('id_test_mask')
        elif task == 'val':
            mask = data.val_mask
        elif task == 'test':
            mask = data.test_mask
        else:
            raise ValueError(f'Task should be train/id_val/id_test/val/test, but got {task}.')
    else:
        mask = ~torch.isnan(data.y)
    if mask is None:
        return None, None
    targets = torch.clone(data.y).detach()
    assert mask.shape[0] == targets.shape[0]
    mask = mask.reshape(targets.shape)
    targets[~mask] = 0

    return mask, targets


def at_stage(i, config):
    r"""
    Test if the current training stage at stage i.

    Args:
        i: Stage that is possibly 1, 2, 3, ...
        config: config object.

    Returns: At stage i.

    """
    if i - 1 < 0:
        raise ValueError(f"Stage i must be equal or larger than 0, but got {i}.")
    if i > len(config.train.stage_stones):
        raise ValueError(f"Stage i should be smaller than the largest stage {len(config.train.stage_stones)},"
                         f"but got {i}.")
    if i - 2 < 0:
        return config.train.epoch < config.train.stage_stones[i - 1]
    else:
        return config.train.stage_stones[i - 2] <= config.train.epoch < config.train.stage_stones[i - 1]



from typing import Union

from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import degree, scatter
from torch_scatter import scatter_mean


def each_homophily(edge_index: Adj, y: Tensor, batch: OptTensor = None,
              method: str = 'edge') -> Union[float, Tensor]:
    assert method in {'edge', 'node', 'edge_insensitive'}
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        if batch is None:
            out = scatter_mean(out, col, 0, dim_size=y.size(0))
            print('out', out.shape)
            return out
        else:
            dim_size = int(batch.max()) + 1
            return scatter(out, batch[col], 0, dim_size)

    elif method == 'node':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
        if batch is None:
            return float(out.mean())
        else:
            return scatter(out, batch, dim=0, reduce='mean')

    elif method == 'edge_insensitive':
        assert y.dim() == 1
        num_classes = int(y.max()) + 1
        assert num_classes >= 2
        batch = torch.zeros_like(y) if batch is None else batch
        num_nodes = degree(batch, dtype=torch.int64)
        num_graphs = num_nodes.numel()
        batch = num_classes * batch + y

        h = each_homophily(edge_index, y, batch, method='edge')
        h = h.view(num_graphs, num_classes)

        counts = batch.bincount(minlength=num_classes * num_graphs)
        counts = counts.view(num_graphs, num_classes)
        proportions = counts / num_nodes.view(-1, 1)

        out = (h - proportions).clamp_(min=0).sum(dim=-1)
        out /= num_classes - 1
        return out if out.numel() > 1 else float(out)

    else:
        raise NotImplementedError

def compute_label_homo(data):

    edge_index, _ = remove_self_loops(data.edge_index)

    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)

    num_labels = torch.max(data.y).item() + 1

    num_nodes = data.y.shape[0]

    row, col = edge_index[0], edge_index[1]

    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes) # 每个节点的度数 deg，即与每个节点相连的边的数量
    degrees_train = deg[data.train_mask]
    degrees_test = deg[data.test_mask]
    # print('test_degree', degrees_test)
    all_degrees_are_one = all(degree == 1 for degree in degrees_test)

    if all_degrees_are_one:
        print("所有节点的度数都是1。")
    else:
        print("不是所有节点的度数都是1。")
    edge_homo_value = (data.y[row] == data.y[col]).int()

    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)

    homo_ratio = torch.squeeze(homo_ratio)

    results = homo_ratio / deg

    return results




class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.00001, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

