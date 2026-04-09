import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def creat_laplace(adj_path):#传入邻接矩阵的路径
    csv_data = pd.read_csv(adj_path)
    n = max(csv_data['from'].max(), csv_data['to'].max()) + 1  # +1 以确保节点编号从 0 到 n-1
    rows, cols, weights = [], [], []
    # 填充邻接矩阵数据和度数向量
    for _, row in csv_data.iterrows():
        i = int(row['from'])
        j = int(row['to'])
        cost = 1
        rows.append(i)
        cols.append(j)
        weights.append(cost)

    # 创建稀疏邻接矩阵（COO -> CSR）
    adj_matrix = sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32)
    grape = (adj_matrix + adj_matrix.T).astype(bool).astype(np.float32)
    grape.setdiag(1)
    # 计算每一行的和（即节点度）
    row_sums = np.array(grape.sum(axis=1)).flatten()  # shape (n,)
    # 构造稀疏对角矩阵
    degrees = sp.diags(row_sums, offsets=0, shape=grape.shape, format='csr', dtype=np.float32)
    # 计算度矩阵的-1/2次方
    degree_values = degrees.diagonal()
    # 计算倒数平方根，避免除以 0
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(degree_values)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # 处理度为0的节点
    # 构造新的稀疏对角矩阵 D^{-1/2}
    degree_inv_sqrt = sp.diags(d_inv_sqrt, offsets=0, shape=degrees.shape, format='csr', dtype=np.float32)
    laplace = degree_inv_sqrt @ grape @ degree_inv_sqrt

    # 输出结果（可选）
    return laplace


if __name__ == '__main__':
    neighbor = 1#告诫邻居
    data_name = 'PEMS03'
    adj_path = os.path.join(root_path, 'data/data', f'{data_name}/{data_name}_adj.csv')
    # 转换为 dense tensor 并计算 neighbor 次幂
    laplace = creat_laplace(adj_path)
    laplace_tensor = torch.tensor(laplace.toarray(), dtype=torch.float32)
    laplace_power6 = torch.matrix_power(laplace_tensor, neighbor)


    # 保存路径
    save_dir = os.path.join(root_path, 'data/data', f'{data_name}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{data_name}_laplace.pt')

    # 保存 tensor
    torch.save(laplace_power6, save_path)
    print(f"✅ laplace^{neighbor} saved to: {save_dir}")