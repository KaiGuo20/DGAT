import pandas as pd
from collections import defaultdict
from argparse import Namespace
from numpy import array
import numpy as np
import os
import re


# df = defaultdict(list)
# for file in os.listdir():
#     if 'GCN_results' not in file:
#         continue
#     if 'Cora-degree-concept' in file:
#         continue
#     with open(file, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             if 'Namespace' in line:
#                 args = eval(line)
#                 df['dataset'].append(args.dataset)
#                 df['model'].append(args.model)
#                 df['seed'].append(args.seed)
#                 df['loss'].append(args.loss)
#
#             if 'Test set results:' in line:
#                 flag = 1
#             if 'Test:' in line:
#                 flag = 2
#             if 'Validatoin:' in line:
#                 flag = 3
#
#             if 'loss= ' in line:
#                 res = line.split()
#                 acc = float(res[-1])
#                 if flag == 1:
#                     df['ori'].append(acc)
#                 if flag == 2:
#                     df['new'].append(acc)
#
# df = pd.DataFrame.from_dict(df, orient='index').transpose()
#
# def fun(x):
#     x = f'{100*x:.2f}'
#     return x
#
# merged = []
# metrics = ['ori', 'new']
# for m in metrics:
#     acc = df.groupby(['dataset', 'model', 'loss'])[m].apply(np.mean)
#     std = df.groupby(['dataset', 'model', 'loss'])[m].apply(np.std).rename('std')
#     # merged += [acc, std]
#     merged += [acc.apply(fun) + '±' + std.apply(fun)]
# new_df = pd.concat(merged, axis=1)
#
#
# # import ipdb
# # ipdb.set_trace()
#
# print(new_df.reset_index().pivot(index=['dataset', 'model'], columns=['loss']))
#
#
# print(new_df)

# data = []
# df = pd.DataFrame()
# folder_path = "/home/guokai/workspace/GOOD1/GCN_results/WebKB-university-covariate"
# for subdir in os.listdir(folder_path):
#     subdir_path = os.path.join(folder_path, subdir)
#     if os.path.isdir(subdir_path):
#         # 遍历子文件夹中的每个文件
#         for file_name in os.listdir(subdir_path):
#             file_path = os.path.join(subdir_path, file_name)
#             if os.path.isfile(file_path) and file_name.endswith('.txt'):
#                 # 读取文件内容
#                 with open(file_path, 'r') as file:
#                     content = file.read()
#                     pattern_mean = r"valid_all_IID-mean (\d+\.\d+)"
#                     match_mean = re.search(pattern_mean, content)
#
#
#                     pattern_IIDtest = r"test_all_IID-mean (\d+\.\d+)"
#                     match_IIDtest = re.search(pattern_IIDtest, content)
#
#                     pattern_IIDOOD = r"test_all_IID-OOD-mean (\d+\.\d+)"
#                     match_IIDOOD = re.search(pattern_IIDOOD, content)
#
#
#                     if match_mean:
#                         a = float(match_mean.group(1))
#                     else:
#                         a = None
#
#                     if match_IIDtest:
#                         b = float(match_IIDtest.group(1))
#                     else:
#                         b = None
#
#                     if match_IIDOOD:
#                         c = float(match_IIDOOD.group(1))
#                     else:
#                         c = None
#
#                     data.append([
#                         subdir,  # 文件夹名称作为指标列的值
#                         a,  # 这里可以是你的实际数值
#                         b,  # 这里可以是你的实际数值
#                         c
#                     ])
#
#                 # 将行添加到数据表格中
# df = pd.DataFrame(data)
# print(df)
# df.to_csv('CSV_results/GCN-WebKB-university-covariate.csv', index=False)
#



data = []
folder_path = "./TTest_results/OOD_APPNP_GATv2_all_5run5"#OOD_ERM_MLP_10run

for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # 遍历子文件夹中的每个文件.
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                # 读取文件内容
                with open(file_path, 'r', encoding='ISO-8859-1') as file:
                # with open(file_path, 'r') as file:
                    content = file.read()
                    pattern_train = r"train_all_IID-mean-final (\d+\.\d+)"
                    matches_train = re.finditer(pattern_train, content)
                    last_train_value = None
                    for match in matches_train:
                        last_train_value = float(match.group(1))  ######取最后一行

                    pattern_mean = r"valid_all_IID-mean-final (\d+\.\d+)"
                    matches_mean = re.finditer(pattern_mean, content)
                    last_mean_value = None
                    for match in matches_mean:
                        last_mean_value = float(match.group(1))######取最后一行
                    # mean_values = []
                    # for match in matches_mean:
                    #     mean_values.append(float(match.group(1)))
                    #
                    # # 获取倒数第二个值
                    # if len(mean_values) >= 2:
                    #     last_mean_value = mean_values[-2]
                    # else:
                    #     last_mean_value = None

                    pattern_test_mean = r"test_all_IID-mean-final (\d+\.\d+)"
                    matches_test_mean = re.finditer(pattern_test_mean, content)
                    last_test_mean_value = None
                    for match in matches_test_mean:
                        last_test_mean_value = float(match.group(1))

                    pattern_OOD_mean = r"test_all_IID-OOD-mean-final (\d+\.\d+)"
                    matches_OOD_mean = re.finditer(pattern_OOD_mean, content)
                    last_OOD_mean_value = None
                    for match in matches_OOD_mean:
                        last_OOD_mean_value = float(match.group(1))
                    # n = 0
                    # for match in matches_OOD_mean:
                    #     n = n+1
                    #     if n == 5:
                    #         last_OOD_mean_value = float(match.group(1))
                    #     else:
                    #         last_OOD_mean_value = None
                    if last_test_mean_value == None:
                        gap = 'None'
                    else:
                        gap = last_test_mean_value - last_OOD_mean_value
                    data.append([
                        subdir,  #
                        last_train_value,
                        last_mean_value,  # valid_all_IID-mean 的最后一行值
                        last_test_mean_value,  # test_all_IID-mean 的最后一行值
                        last_OOD_mean_value,  # test_all_IID-OOD-mean 的最后一行值
                        gap
                    ])

df = pd.DataFrame(data, columns=['parameter','train_all_IID-mean', 'valid_all_IID-mean', 'test_all_IID-mean', 'test_all_IID-OOD-mean', 'gap'])
print(df)
df.to_csv('CSV_TTest_results/OOD_APPNP_GATv2_all_5run5.csv', index=False)
