import csv

import os
import sys

import numpy as np
import pandas as pd

import torch
print(sys.version)

"""root_path = os.path.dirname(os.path.abspath(__file__))
data_name = 'PEMS03'
data_path = os.path.join(root_path, 'data/data', data_name, f'{data_name}.csv')
print(data_path)
data_csv = pd.read_csv(data_path)
print(data_csv.columns)

max_node = data_csv['from'].max()
min_node = data_csv['from'].min()
print(max_node, min_node)

data_list = []
for i in data_csv['from']:
    data_list.append(i)
for i in data_csv['to']:
    data_list.append(i)
data_list.sort()
data_list = list(set(data_list))
print(len(data_list))
data_value = {value: index for index, value in enumerate(data_list)}
print(data_value)

data_npz = np.load(r"D:\CodeProject\Python\traffic\traffic_prediction1\data\data\PEMS03\PEMS03.npz")['data']
print(data_npz.shape)

out_list = [['from', 'to', 'distance']]
for fro, to, dis in zip(data_csv['from'], data_csv['to'], data_csv['distance']):
    i = data_value[fro]
    j = data_value[to]
    out_list.append([i, j, dis])

print(out_list)

# 打开（或创建）CSV 文件进行写入
with open(r'D:\CodeProject\Python\traffic\traffic_prediction1\data\data\PEMS03\PEMS03_adj.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(out_list)  # 将数据写入文件"""