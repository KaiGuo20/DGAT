import pandas as pd
from collections import defaultdict
from argparse import Namespace
from numpy import array
import numpy as np
import os
import re



data = []
folder_path = "./TTest_results/OOD_ERM_MLP_10run"

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

                    # match = re.search(r"Gap-final \[([^]]+)\]", content)
                    # GAP_final_list = None
                    # if match:
                    #     content = match.group(1)  # 提取匹配的内容
                    #     # 将匹配的内容转换为 Python 列表
                    #     GAP_final_list = [float(item.strip()) for item in content.split(',')]


                    pattern_test_mean = r"Gap-final-mean (\d+\.\d+)"
                    matches_test_mean = re.finditer(pattern_test_mean, content)
                    last_test_mean_value = None
                    for match in matches_test_mean:
                        last_test_mean_value = float(match.group(1))

                    pattern_OOD_mean = r"test_all_IID-OOD-mean-final (\d+\.\d+)"
                    matches_OOD_mean = re.finditer(pattern_OOD_mean, content)
                    last_OOD_mean_value = None
                    for match in matches_OOD_mean:
                        last_OOD_mean_value = float(match.group(1))

                    match = re.search(r"test_all_IID_OOD-final \[([^]]+)\]", content)
                    last_OOD_mean_list = None
                    if match:
                        content = match.group(1)  # 提取匹配的内容
                        # 将匹配的内容转换为 Python 列表
                        last_OOD_mean_list = [float(item.strip()) for item in content.split(',')]

                    data.append([
                        subdir,  # 文件夹名称作为指标列的值
                        # GAP_final_list,  # valid_all_IID-mean 的最后一行值
                        last_test_mean_value,  # test_all_IID-mean 的最后一行值
                        last_OOD_mean_value,  # test_all_IID-OOD-mean 的最后一行值
                        last_OOD_mean_list
                    ])

df = pd.DataFrame(data, columns=['GAP_final_list', 'GAP_final_mean', 'test_all_IID-OOD-mean', 'test_all_IID-OOD-final'])
print(df)
df.to_csv('CSV_TTest_results/OOD_ERM_MLP_10run.csv', index=False)
