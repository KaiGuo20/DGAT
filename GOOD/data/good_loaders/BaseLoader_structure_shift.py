import random
import dgl
import pandas
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler
from GOOD.utils.CMD import cmd, CMD, mmd
import torch.nn.functional as F

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.data.good_datasets.data_generate import _compute_density_property, _compute_locality_property, _compute_popularity_property
from typing import List, Iterator
from torch.utils.data.sampler import Sampler
from GOOD.utils.node_property_split import NodePropertySplit
from torch_geometric.data.dataset import Dataset
from torch_geometric.data import Data
from GOOD.utils.train import each_homophily,compute_label_homo
import numpy as np
import torch
from torch_geometric.utils import homophily
from torch_geometric.utils import subgraph
from GOOD.utils.PE import laplacian_positional_encoding, re_features
from torch_geometric.utils import to_torch_coo_tensor
def calculate_label_distribution(labels, mask):
    masked_labels = labels[mask]
    num_classes = np.max(labels) + 1  # 类别数量
    class_counts = np.zeros(num_classes)  # 用于统计每个类别的样本数量

    for i in range(num_classes):
        class_counts[i] = np.sum(masked_labels == i)

    class_distribution = class_counts / np.sum(class_counts)
    # class_distribution = ["{:.0f}".format(num) for num in class_distribution]
    # class_distribution = [float(num.replace("'", "")) for num in class_distribution]
    return class_distribution

def compare_class_distributions(labels, mask1, mask2):
    # 根据掩码计算分布
    dist1 = calculate_label_distribution(labels, mask1)#+1e-20
    dist2 = calculate_label_distribution(labels, mask2)#+1e-20
    print('IID-test',dist1)
    print('OOD-test',dist2)
    # 计算 KL 散度
    kl_divergence = sum(dist1[label] * np.log(dist1[label] / dist2[label])
                        for label in range(len(dist1)) if label < len(dist2))

    return kl_divergence
@register.dataloader_register
class BaseDataLoader1(Munch):

    def __init__(self, *args, **kwargs):
        super(BaseDataLoader1, self).__init__(*args, **kwargs)

    @classmethod
    def setup(cls, dataset, config: Union[CommonArgs, Munch]):
        r"""
        Create a PyG data loader.

        Args:
            dataset: A GOOD dataset.
            config: Required configs:
                ``config.train.train_bs``
                ``config.train.val_bs``
                ``config.train.test_bs``
                ``config.model.model_layer``
                ``config.train.num_steps(for node prediction)``

        Returns:
            A PyG dataset loader.

        """
        reset_random_seed(config)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(config.random_seed)

        if config.model.model_level == 'node':
            if config.dataset.dataset_name == "GOODPubmed" or config.dataset.dataset_name == "GOODCiteseer":
                graph = dataset
            else:
                graph = dataset[0]
            import networkx as nx
            import matplotlib.pyplot as plt
##########################################################3
            # # 假设有一个 edge_index
            # half_edges = graph.edge_index[:, :graph.edge_index.size(1) // 16]
            #
            # # 创建一个wu向图
            # G = nx.Graph()
            #
            # # 添加前一半边到图中
            # edges = graph.edge_index.numpy().T.tolist()
            # half_train_mask = graph.train_mask[:len(half_edges[0])]
            # half_test_mask = graph.test_mask[:len(half_edges[0])]
            # G.add_edges_from(edges)
            #
            # # 根据train mask和test mask挑选出相应的节点
            # train_nodes = [i for i, mask in enumerate(graph.train_mask) if mask]
            # test_nodes = [i for i, mask in enumerate(graph.test_mask) if mask]
            #
            # # 使用nx.draw绘制图形，分别设置训练集和测试集节点的颜色
            # pos = nx.spring_layout(G)  # 为了更好的布局，使用 spring_layout
            # # nx.draw(G, pos, font_weight='bold', node_color='lightblue', arrowsize=1, node_size=2)
            # nx.draw(G, pos, nodelist=train_nodes, node_color='red', arrowsize=1, node_size=2)
            # nx.draw(G, pos, nodelist=test_nodes, node_color='green', arrowsize=1, node_size=2)
            #
            # # # 将节点分为训练集和测试集
            # # train_nodes = [i for i, mask_value in enumerate(graph.train_mask) if mask_value]
            # # test_nodes = [i for i, mask_value in enumerate(graph.test_mask) if mask_value]
            # print('1111')
            # # # 绘制图，标记训练集和测试集的节点
            # # pos = nx.spring_layout(G)  # 为了更好的布局，可以调整布局算法
            # # nx.draw(G, pos, with_labels=True, font_weight='bold')
            # nx.draw(G, with_labels=False, font_weight='bold', node_color='blue', arrowsize=1, node_size=2)
            # plt.show

            # 标记训练集和测试集的节点
            # nx.draw_networkx_nodes(G, pos, nodelist=train_nodes, node_color='r', label='Train Set')
            # nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, node_color='b', label='Test Set')

            # # plt.legend()
            # save_path = "/mnt/home/guokai1/workspace/GOOD/figure/arxiv_degree_concept_graph.png"
            # plt.savefig(save_path)
##################################GOODArxiv##################################
            if config.dataset.dataset_name == "GOODArxiv":
                print('----------arxiv---------')
                property_name = 'density'
                ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
                transform1 = NodePropertySplit(property_name, ratios)
                print(transform1)
                _graph = Data(edge_index=graph.edge_index, num_nodes=graph.x.size()[0])
                data_new = transform1(_graph)
                graph.train_mask = data_new.id_train_mask
                graph.id_val_mask = data_new.id_val_mask
                graph.id_test_mask = data_new.id_test_mask
                graph.val_mask = data_new.ood_val_mask
                graph.test_mask = data_new.ood_test_mask
                print('---data1', data_new)

                print('graph.y', graph.y)

                kl_divergence1 = compare_class_distributions(graph.y.numpy(), graph.train_mask, graph.id_test_mask)
                kl_divergence2 = compare_class_distributions(graph.y.numpy(), graph.train_mask, graph.test_mask)
                print("KL Divergence between IID and OOD distributions:", kl_divergence1)
                print("KL Divergence between IID and OOD distributions:", kl_divergence2)
                train_dis = calculate_label_distribution(graph.y.numpy(), graph.train_mask)
                IID_dis = calculate_label_distribution(graph.y.numpy(), graph.id_test_mask)
                OOD_dis = calculate_label_distribution(graph.y.numpy(), graph.test_mask)
                print(train_dis)
                print(IID_dis)
                print(OOD_dis)
                if config.model.feature_type == 'raw':
                    print('raw')
                    data = graph
                    print(data)
                if config.model.feature_type == 'sbert':
                    print('original_x', graph.x.size())
                    data = torch.load(
                        f"/mnt/home/guokai1/workspace/GOOD/GOOD/networks/models/{'arxiv'}_{'fixed'}_{'sbert'}.pt",
                        map_location='cpu')
                    print('--data', data)
                    graph.x = data.x
                    print('---------x', graph.x.size())
                if config.model.feature_type == 'e5':
                    print('original_x', graph.x.size())
                    data = torch.load(
                        f"/mnt/home/guokai1/workspace/GOOD/GOOD/networks/models/{'arxiv'}_{'fixed'}_{'e5'}.pt",
                        map_location='cpu')
                    graph.x = data.x
                    print('x', graph.x.size())
                # train = F.normalize(graph.x[graph.train_mask], dim=-1)
                # test = F.normalize(graph.x[graph.test_mask], dim=-1)
                # # train = graph.x[graph.train_mask]
                # # test = graph.x[graph.test_mask]
                # cmd_value = mmd(train, test)
                # print('mmd_value------------------', cmd_value)
                ###################################################
                # import networkx as nx
                # import matplotlib.pyplot as plt
                #
                # # 假设有一个 edge_index
                # edge_index = graph.edge_index
                # G = nx.Graph()
                #
                # # 添加边到图中
                # G.add_edges_from(edge_index.t().tolist())
                #
                # # 将节点分为训练集和测试集
                # train_nodes = [i for i, mask_value in enumerate(graph.train_mask) if mask_value]
                # test_nodes = [i for i, mask_value in enumerate(graph.test_mask) if mask_value]
                # print('1111')
                # # 绘制图，标记训练集和测试集的节点
                # pos = nx.spring_layout(G)  # 为了更好的布局，可以调整布局算法
                # nx.draw(G, pos, with_labels=True, font_weight='bold')
                #
                # # 标记训练集和测试集的节点
                # nx.draw_networkx_nodes(G, pos, nodelist=train_nodes, node_color='r', label='Train Set')
                # nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, node_color='b', label='Test Set')
                #
                # plt.legend()
                # save_path = "/mnt/home/guokai1/workspace/GOOD/figure/arxiv_degree_concept_graph.png"
                # plt.savefig(save_path)
                ###################################################################
                from torch_geometric.utils import to_networkx
                import networkx as nx
                import matplotlib.pyplot as plt
                edge_index_tra, _ = subgraph(
                    subset=graph.train_mask,
                    edge_index=graph['edge_index'],
                    relabel_nodes=True
                )
                edge_index_test, _ = subgraph(
                    subset=graph.test_mask,
                    edge_index=graph.edge_index,
                    relabel_nodes=True
                )
                print('train_mask_shape',graph.train_mask.shape)
                print('test_mask_shape', graph.test_mask.shape)
                count_true = torch.sum(graph.test_mask).item()

                print("True的个数：", count_true)
                print('mask_test_edge', edge_index_tra)
                _graph = Data(edge_index=graph.edge_index, num_nodes=data.x.size()[0])
                _graph_ = to_networkx(_graph)
                degrees = dict(_graph_.degree())
                # 将字典的键和值转换为列表
                keys_list = list(degrees.keys())
                values_list = list(degrees.values())

                # 使用掩码张量获取满足条件的键和值
                selected_keys_tra = [key for key, m in zip(keys_list, graph.train_mask) if m]
                selected_values_tra = [value for value, m in zip(values_list, graph.train_mask) if m]

                # 将选定的键和值重新组合成字典
                degrees_train = dict(zip(selected_keys_tra, selected_values_tra))
                selected_keys_test = [key for key, m in zip(keys_list, graph.test_mask) if m]
                selected_values_test = [value for value, m in zip(values_list, graph.test_mask) if m]

                # 将选定的键和值重新组合成字典
                degrees_test = dict(zip(selected_keys_test, selected_values_test))
#########################################################################
                # from torch_geometric.utils import degree
                # import matplotlib.pyplot as plt
                #
                # # 使用 degree 函数计算每个节点的度数
                # degrees = degree(graph.edge_index[0], num_nodes=data.x.size()[0], dtype=torch.long)
                #
                # print("每个节点的度数:", degrees)
                # degrees_train = degrees[graph.train_mask]
                # degrees_test = degrees[graph.test_mask]
                # print('test_degree', degrees_test)
                # all_degrees_are_one = all(degree == 1 for degree in degrees_test)
                #
                # if all_degrees_are_one:
                #     print("所有节点的度数都是1。")
                # else:
                #     print("不是所有节点的度数都是1。")


                #绘制散点图
                # plt1 = plt.figure()
                # plt.scatter(list(degrees_train.keys()), list(degrees_train.values()), color='blue', label='Train Set', alpha=0.7)
                # plt.scatter(list(degrees_test.keys()), list(degrees_test.values()), color='orange', label='Test Set', alpha=0.7)
                # plt.ylim(bottom=1)
                # plt.title("Node Degree Scatter Plot")
                # plt.xlabel("Node")
                # plt.ylabel("Degree")
                # plt.legend()
                # save_path = "/mnt/home/guokai1/workspace/GOOD/figure/arxiv_time_concept_new.png"
                # plt1.savefig(save_path)

                # 获取度数的计数
                from collections import Counter
                count_train = Counter(degrees_train.values())
                count_test = Counter(degrees_test.values())

                # 将计数排序为度数顺序
                degrees_train, count_train = zip(*sorted(count_train.items()))
                degrees_test, count_test = zip(*sorted(count_test.items()))

                # 计算密度
                total_nodes_train = sum(count_train)
                total_nodes_test = sum(count_test)

                density_train = [count / total_nodes_train for count in count_train]
                density_test = [count / total_nodes_test for count in count_test]

                # 绘制散点图
                plt2 = plt.figure()
                plt.scatter(degrees_train, density_train, color='blue', label='Train Set', alpha=0.7)
                plt.scatter(degrees_test, density_test, color='orange', label='Test Set', alpha=0.7)
                # plt.xlim(0, 5000)  # 设置 x 轴的范围
                plt.xlabel('Degree')
                plt.ylabel('Density')
                plt.legend()
                # save_path = "/mnt/home/guokai1/workspace/GOOD/figure/arxiv_time_concept_density_new.png"
                # plt2.savefig(save_path)
##################################GOODArxiv##################################
########################################################
            ####################################################
            if config.model.model_name == 'GTransformer' or config.model.model_name == 'GTGCN':
                row, col = graph.edge_index
                g = dgl.graph((row, col))
                g = dgl.to_bidirected(g)
                lpe = laplacian_positional_encoding(g, config.model.pe_dim)

                graph.x = torch.cat((graph.x, lpe), dim=1)
                # dataset[0].edge_index = to_torch_coo_tensor(dataset[0].edge_index)
                x = re_features(graph.edge_index, graph.x, config.model.hops)

                graph.x = x

            if config.model.model_name == 'GPS'or config.model.model_name == 'GTV2':
                row, col = graph.edge_index
                g = dgl.graph((row, col))
                g = dgl.to_bidirected(g)
                lpe = laplacian_positional_encoding(g, config.model.pe_dim)

                graph.x = torch.cat((graph.x, lpe), dim=1)
            ##############################################################
            loader = GraphSAINTRandomWalkSampler(graph, batch_size=config.train.train_bs,
                                                 walk_length=config.model.model_layer,
                                                 num_steps=config.train.num_steps, sample_coverage=100,
                                                 save_dir=dataset.processed_dir)
            # loader = graph
            # print('-----------------graph---', graph)
            if config.ood.ood_alg == 'EERM':
                loader = {'train': [graph], 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
            else:
                loader = {'train': loader, 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
            # else:
            #     loader = {'train': [graph], 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
            #               'test': [graph]}

        else:
            loader = {'train': DataLoader(dataset['train'], batch_size=config.train.train_bs, shuffle=True, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'eval_train': DataLoader(dataset['train'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'id_val': DataLoader(dataset['id_val'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g) if dataset.get(
                          'id_val') else None,
                      'id_test': DataLoader(dataset['id_test'], batch_size=config.train.test_bs,
                                            shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g) if dataset.get(
                          'id_test') else None,
                      'val': DataLoader(dataset['val'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'test': DataLoader(dataset['test'], batch_size=config.train.test_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g)}

        # print('-----------graph', dataset[0])
        # # import ipdb;ipdb.set_trace()
        # print('-----------dataset', dataset)


        #
        if config.dataset.dataset_name == "GOODPubmed" or config.dataset.dataset_name == "GOODCiteseer":
            graph = dataset
        else:
            graph = dataset[0]
        train_mask = graph['train_mask']
        test_mask = graph['test_mask']
        # edge_index_tra = graph['edge_index']
        edge_index_tra, _ = subgraph(
            subset=graph.train_mask,
            edge_index=graph['edge_index'],
            relabel_nodes=True
        )
        edge_index_test, _ = subgraph(
            subset=graph.test_mask,
            edge_index=graph['edge_index'],
            relabel_nodes=True
        )
        # edge_index = graph.edge_index
        # edge_index_test = edge_index[graph.test_mask]
        # print('graph.test_mask', graph['test_mask'])
        # print('graph.train_mask', graph.train_mask)
        labels_tra = graph['y'][train_mask]
        labels_test = graph['y'][test_mask]
        # print('labels_test',labels_test)
        # print('labels_tra', labels_tra)
        # print('edge_index_tra', edge_index_tra)
        # print('edge_index_test', edge_index_test)
        # edge_index_test = edge_index_test.to(torch.int64)
        # edge_index_tra = edge_index_tra.to(torch.int64)
        labels_tra = labels_tra.to(torch.int64)
        labels_test = labels_test.to(torch.int64)


        # homophily_train = each_homophily(edge_index_tra, labels_tra, method='edge')
        # homophily_test = each_homophily(edge_index_test, labels_test, method='edge')
        # print('homo_train',homophily_train.mean())
        # print('homo_test', homophily_test.mean())
        eachnode_homophily = compute_label_homo(graph)
        print('homo---', eachnode_homophily)
        homophily_train = eachnode_homophily[graph.train_mask].mean()
        homophily_test = eachnode_homophily[graph.test_mask].mean()
        print('homo',homophily_train)
        print('homo', homophily_test)
        # homophily1 = each_homophily(graph.edge_index, graph.y, method='edge')
        # homophily_train = homophily1[graph.train_mask]
        # homophily_test = homophily1[graph.test_mask]
        # print('homo_train',homophily_train.mean())
        # print('homo_test', homophily_test.mean())
        return cls(loader)