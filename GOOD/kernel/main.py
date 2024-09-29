r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union

import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.definitions import OOM_CODE
import numpy as np
import yaml
import random
from ruamel.yaml import YAML


def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)
    # reset_random_seed(seed)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    if config.dataset.dataset_name == "pubmed":
        dataset = dataset
    print('dataset----------')
    print(f"#D#Dataset: {dataset}")
    # print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader


def main():
    # yaml = YAML
    run_num = 2
    train_all_IID, valid_all_IID, test_all_IID, OOD_all_test = [], [], [], []
    train_all_OOD, valid_all_OOD, test_all_OOD = [], [], []
    # seeds = [0, 1, 2, 3, 4]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(run_num):
        #########################################################################
        seed = seeds[i]
        # script_dir = '/home/guokai/workspace/GOOD1/configs/GOOD_configs/'
        # yaml_file_path = os.path.join(script_dir, 'base.yaml')
        # print(yaml_file_path)
        # with open(yaml_file_path, 'r', encoding='utf-8') as f:
        #     config1 = yaml.safe_load(f)
        #
        # # 修改YAML配置值
        # print(config1)
        # config1['random_seed'] = seed  # 设置新的种子值
        # print(config1)
        #
        # # 将修改后的配置保存回YAML文件
        # with open(yaml_file_path, 'w', encoding='utf-8') as f:
        #     yaml.safe_dump(config1, f, default_flow_style=False)
        args = args_parser()
        config = config_summoner(args)
        config.random_seed = seed
        print('main_seed', config.random_seed)
################################################3
        if config.train.type == 'Init':
            print('----------------MLP model')
            ### MLP model

            model_name = config.model.model_name
            config.model.model_name = 'MLP'
            epoch = config.train.max_epoch
            config.train.max_epoch = 10
            # load_logger(config)
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task()

            if config.task == 'train':
                pipeline.task = 'test'
                train_IID, valid_IID, test_IID, OOD_test, train_OOD, valid_OOD, test_OOD = pipeline.load_task()

                train_all_IID.append(train_IID)
                valid_all_IID.append(valid_IID)
                test_all_IID.append(test_IID)
                OOD_all_test.append(OOD_test)
                train_all_OOD.append(train_OOD)
                valid_all_OOD.append(valid_OOD)
                test_all_OOD.append(test_OOD)
                print('train_all_IID', train_all_IID)
                print('test_all_IID_OOD', OOD_all_test)
                print('test_all_IID-OOD-mean', np.mean(OOD_all_test))
                print('test_all_IID-OOD-std', np.std(OOD_all_test, ddof=1))

                print('train_all_OOD', train_all_OOD)
                print('test_all_OOD', test_all_OOD)
                print('test_all_OOD-mean', np.mean(test_all_OOD))
                print('test_all_OOD-std', np.std(test_all_OOD, ddof=1))

                print(f"saving model_mlpinit at epcoh {epoch}")
                torch.save(model.state_dict(), f"./model_mlpinit.pt")

                ####GNN model
                config.model.model_name = model_name
                config.train.max_epoch = epoch
                model, loader = initialize_model_dataset(config)

                model.load_state_dict(torch.load(f"./model_mlpinit.pt"))########load MLP init
                ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
                pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
                pipeline.load_task()

                if config.task == 'train':
                    pipeline.task = 'test'
                    train_IID, valid_IID, test_IID, OOD_test, train_OOD, valid_OOD, test_OOD = pipeline.load_task()

                    train_all_IID.append(train_IID)
                    valid_all_IID.append(valid_IID)
                    test_all_IID.append(test_IID)
                    OOD_all_test.append(OOD_test)
                    train_all_OOD.append(train_OOD)
                    valid_all_OOD.append(valid_OOD)
                    test_all_OOD.append(test_OOD)
                    print('train_all_IID', train_all_IID)
                    print('train_all_IID-mean', np.mean(train_all_IID))
                    print('train_all_IID-std', np.std(train_all_IID, ddof=1))
                    print('valid_all_IID', valid_all_IID)
                    print('valid_all_IID-mean', np.mean(valid_all_IID))
                    print('valid_all_IID-std', np.std(valid_all_IID, ddof=1))
                    print('test_all_IID', test_all_IID)
                    print('test_all_IID-mean', np.mean(test_all_IID))
                    print('test_all_IID-std', np.std(test_all_IID, ddof=1))
                    print('test_all_IID_OOD', OOD_all_test)
                    print('test_all_IID-OOD-mean', np.mean(OOD_all_test))
                    print('test_all_IID-OOD-std', np.std(OOD_all_test, ddof=1))

                    print('train_all_OOD', train_all_OOD)
                    print('train_all_OOD-mean', np.mean(train_all_OOD))
                    print('train_all_OOD-std', np.std(train_all_OOD, ddof=1))
                    print('valid_all_OOD', valid_all_OOD)
                    print('valid_all_OOD-mean', np.mean(valid_all_OOD))
                    print('valid_all_OOD-std', np.std(valid_all_OOD, ddof=1))
                    print('test_all_OOD', test_all_OOD)
                    print('test_all_OOD-mean', np.mean(test_all_OOD))
                    print('test_all_OOD-std', np.std(test_all_OOD, ddof=1))
            model.to('cpu')
            # 训练结束后释放显存
            del model
            del loader
            del args
            del config
            del pipeline
            del ood_algorithm
            # del optimizer
            # del loss

            # 将所有相关变量置为None
            model = None
            # optimizer = None
            # loss = None

            torch.cuda.empty_cache()
        else:
            # args = args_parser()
            # config = config_summoner(args)
            # config.random_seed = seed
            # print('main_seed', config.random_seed)
            # load_logger(config)
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task()

            if config.task == 'train':
                pipeline.task = 'test'
                train_IID, valid_IID, test_IID , OOD_test, train_OOD, valid_OOD, test_OOD = pipeline.load_task()

                train_all_IID.append(train_IID)
                valid_all_IID.append(valid_IID)
                test_all_IID.append(test_IID)
                OOD_all_test.append(OOD_test)
                train_all_OOD.append(train_OOD)
                valid_all_OOD.append(valid_OOD)
                test_all_OOD.append(test_OOD)
                print('train_all_IID', train_all_IID)
                print('train_all_IID-mean', np.mean(train_all_IID))
                print('train_all_IID-std', np.std(train_all_IID, ddof=1))
                print('valid_all_IID', valid_all_IID)
                print('valid_all_IID-mean', np.mean(valid_all_IID))
                print('valid_all_IID-std', np.std(valid_all_IID, ddof=1))
                print('test_all_IID', test_all_IID)
                print('test_all_IID-mean', np.mean(test_all_IID))
                print('test_all_IID-std', np.std(test_all_IID, ddof=1))
                print('test_all_IID_OOD', OOD_all_test)
                print('test_all_IID-OOD-mean', np.mean(OOD_all_test))
                print('test_all_IID-OOD-std', np.std(OOD_all_test, ddof=1))

                print('train_all_OOD', train_all_OOD)
                print('train_all_OOD-mean', np.mean(train_all_OOD))
                print('train_all_OOD-std', np.std(train_all_OOD, ddof=1))
                print('valid_all_OOD', valid_all_OOD)
                print('valid_all_OOD-mean', np.mean(valid_all_OOD))
                print('valid_all_OOD-std', np.std(valid_all_OOD, ddof=1))
                print('test_all_OOD', test_all_OOD)
                print('test_all_OOD-mean', np.mean(test_all_OOD))
                print('test_all_OOD-std', np.std(test_all_OOD, ddof=1))
            model.to('cpu')
            # 训练结束后释放显存
            del model
            del loader
            del args
            del config
            del pipeline
            del ood_algorithm
            # del optimizer
            # del loss

            # 将所有相关变量置为None
            model = None
            # optimizer = None
            # loss = None

            torch.cuda.empty_cache()


    print('train_all_IID', train_all_IID)
    print('train_all_IID-mean-final', np.mean(train_all_IID))
    print('train_all_IID-std-final', np.std(train_all_IID, ddof=1))
    print('valid_all_IID-final', valid_all_IID)
    print('valid_all_IID-mean-final', np.mean(valid_all_IID))
    print('valid_all_IID-std-final', np.std(valid_all_IID, ddof=1))
    print('test_all_IID-final', test_all_IID)
    print('test_all_IID-mean-final', np.mean(test_all_IID))
    print('test_all_IID-std-final', np.std(test_all_IID, ddof=1))
    print('test_all_IID_OOD-final', OOD_all_test)
    print('test_all_IID-OOD-mean-final', np.mean(OOD_all_test))
    print('test_all_IID-OOD-std-final', np.std(OOD_all_test, ddof=1))
    gap_final = torch.tensor(test_all_IID) - torch.tensor(OOD_all_test)
    print('Gap-final', gap_final.tolist())
    print('Gap-final-mean', np.mean(gap_final.tolist()))
    print('Gap-final-std', np.std(gap_final.tolist(), ddof=1))

    print('train_all_OOD', train_all_OOD)
    print('train_all_OOD-mean-final', np.mean(train_all_OOD))
    print('train_all_OOD-std-final', np.std(train_all_OOD, ddof=1))
    print('valid_all_OOD', valid_all_OOD)
    print('valid_all_OOD-mean-final', np.mean(valid_all_OOD))
    print('valid_all_OOD-std-final', np.std(valid_all_OOD, ddof=1))
    print('test_all_OOD', test_all_OOD)
    print('test_all_OOD-mean-final', np.mean(test_all_OOD))
    print('test_all_OOD-std-final', np.std(test_all_OOD, ddof=1))

def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()
