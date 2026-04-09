import argparse

import torch

from traffic_prediction1.model.train import train_
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--pems', type=str, default='PEMS08',
                        help='PEMS04, PEMS08, METR_LA')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test data ratio')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch', type=int, default=50, help='batch size')
    parser.add_argument('--log_file', default=os.path.join(PROJECT_ROOT, r'data/log'), help='log file')  # 日志文件
    parser.add_argument('--prediction_time', type=int, default=60, help='15, 30, 60')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--weights_file', type=str,
                        default=os.path.join(PROJECT_ROOT, r'data/weights_file'))
    parser.add_argument('--device', type=str, default='cuda', help='CPU or cuda')
    parser.add_argument('--file_path', default=PROJECT_ROOT, help='file_path')  # 日志文件
    parser.add_argument('--wait', type=int, default=15)
    parser.add_argument('--pre_training', type=bool, default=False, help='start pre-training')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_(args)





if __name__ == '__main__':
    main()

