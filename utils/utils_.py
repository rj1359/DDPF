import argparse
import datetime
import os
import random
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def RMSE_loss(y_pred, y_true):
    """
    均方根误差 (RMSE) 损失函数
    """
    # 计算均方误差 (MSE)
    mse = torch.mean((y_true - y_pred) ** 2)
    # 计算RMSE，返回结果是误差的平方根
    rmse = torch.sqrt(mse)
    return rmse

def MAPE_loss(y_pred, y_true, epsilon=1):
    """
    平滑版 MAPE 损失函数，避免除以接近0的 y_true 导致爆炸。
    """
    denominator = torch.clamp(torch.abs(y_true), min=epsilon)
    loss = torch.abs((y_true - y_pred) / denominator)
    return torch.mean(loss) * 100  # 返回百分比误差


def MAE_loss(y_pred, y_true):
    # 确保在同一设备
    y_true = y_true.to(y_pred.device)

    # 计算平均绝对误差，使用 PyTorch 的向量化操作
    loss = torch.abs(y_true - y_pred)  # 计算绝对误差
    mean_loss = loss.mean()  # 计算所有元素的平均绝对误差

    return mean_loss


def log_write(log_path, information):
    with open(log_path, 'a') as f:
        f.write(information + '\n')
        f.flush()
        print(information)

def read_data(data, index, pre_step, std_mean):
    x = np.asarray(data[index:index+12, :, :])
    #label = np.asarray(data[index+23+pre_step, :, :])
    label = np.asarray(data[index+12:index + 12 + pre_step, :, :])
    x = torch.from_numpy(x).float()
    label = torch.from_numpy(label).float()
    for i in range(len(std_mean)-1):
        x[:, :, i+1] = (x[:, :, i+1] - std_mean[i+1][1]) / std_mean[i+1][0]
    for i in range(len(std_mean)-1):
        label[:, :, i + 1] = (label[:, :, i + 1] - std_mean[i+1][1]) / std_mean[i+1][0]
    #return x, label.unsqueeze(0)  # label shape: [1, 307, 1]
    return x, label

class TrafficDataset(Dataset):
    def __init__(self, data, std_mean, prediction_time):
        self.data = data
        self.pred_step = int(prediction_time / 5)
        self.std_mean = std_mean
        self.index = list(range(len(data) - 24 - self.pred_step))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x, label = read_data(self.data, idx, self.pred_step, self.std_mean)
        return x, label


def read_data1(data, index, pre_step, std_mean):
    x = np.asarray(data[index:index+12, :, :])
    #label = np.asarray(data[index+23+pre_step, :, :])
    label = np.asarray(data[index+12:index + 12 + pre_step, :, :])
    x = torch.from_numpy(x).float()
    label = torch.from_numpy(label).float()
    for i in range(len(std_mean)):
        x[:, :, i] = (x[:, :, i] - std_mean[i][1]) / std_mean[i][0]
    for i in range(len(std_mean)-1):
        label[:, :, i + 1] = (label[:, :, i + 1] - std_mean[i+1][1]) / std_mean[i+1][0]
    #return x, label.unsqueeze(0)  # label shape: [1, 307, 1]
    return x, label

class TrafficDataset1(Dataset):
    def __init__(self, data, std_mean, prediction_time):
        self.data = data
        self.pred_step = int(prediction_time / 5)
        self.std_mean = std_mean
        self.index = list(range(len(data) - 24 - self.pred_step))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x, label = read_data1(self.data, idx, self.pred_step, self.std_mean)
        return x, label



def load_data(args):
    # 加载 .npz 文件，autodl-tmp/traffic \root\autodl-tmp
    PROJECT_ROOT = args.file_path
    #PROJECT_ROOT = r'D:\CodeProject\Python\traffic\traffic_prediction1'
    pems = [os.path.join(PROJECT_ROOT, r"data/data/PEMS03/PEMS03.npz"),
            os.path.join(PROJECT_ROOT, r"data/data/PEMS04/PEMS04.npz"),
            os.path.join(PROJECT_ROOT, r"data/data/PEMS08/PEMS08.npz"),
            os.path.join(PROJECT_ROOT, r"data/data/METR_LA/METR_LA.npz"),
            ]
    pems_start_time = ['2018-9-1', '2018-1-1', '2016-7-1', '2012-3-1']
    pems_data = None
    laplace, std_mean = None, None

    for file_index in range(len(pems)):
        if os.path.splitext(os.path.basename(pems[file_index]))[0] == args.pems:
            print("数据路径:", pems[file_index])
            pems_data = np.load(pems[file_index]).get('data').astype(np.float32)
            pems_week_data =np.zeros((pems_data.shape[0], pems_data.shape[1], 1))
            pems_day_data = np.zeros((pems_data.shape[0], pems_data.shape[1], 1))
            start_time = datetime.datetime.strptime(pems_start_time[file_index], "%Y-%m-%d")
            for day in range(int(pems_data.shape[0]/288)):
                pems_week_data[day*288:(day+1)*288, :, :] = np.full((288, pems_data.shape[-2], 1), (start_time + datetime.timedelta(days=day)).weekday())
            for minute in range(pems_data.shape[0]):
                pems_day_data[minute, :, :] = minute % 288
            pems_data = pems_data[:, :, 0:1]
            pems_data = np.concatenate([pems_data, pems_week_data, pems_day_data], axis=2)

            print("加载laplace矩阵")
            laplace_path = os.path.join(PROJECT_ROOT, f"data/data/{args.pems}/{args.pems}_laplace.pt")
            laplace = torch.load(laplace_path)

    train_step = int(pems_data.shape[0]*args.train_ratio)
    val_step = int(pems_data.shape[0]*args.val_ratio)
    train = pems_data[:train_step, :, :]
    val = pems_data[train_step:train_step+val_step, :, :]
    test = pems_data[train_step+val_step:, :, :]

    std_mean = np.zeros((pems_data.shape[-1], 2), dtype=np.float32)
    for i in range(pems_data.shape[-1]):
        std_mean[i, 0] = np.std(train[:, :, i])
        std_mean[i, 1] = np.mean(train[:, :, i])
    #(11894, 307, 5) (1699, 307, 5) (3399, 307, 5)
    print(f'train shape: {train.shape}, val shape: {val.shape}, test shape: {test.shape}')
    return train, val, test, laplace, std_mean



if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pems', type=str, default='PEMS04',
                        help='PEMS03, PEMS04, PEMS07, PEMS08')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test data ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--prediction_time', type=int, default=30, help='15, 30, 60')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--file_path', default=PROJECT_ROOT, help='file_path')  # 日志文件
    args = parser.parse_args()
    if args.mode == 'train':
        args.prediction_time = 5
    train, val, test, laplace, std_mean = load_data(args)
    train_loader = DataLoader(TrafficDataset(train, std_mean, args.prediction_time),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)

