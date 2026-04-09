from traffic_prediction1.model.model3 import Model
from traffic_prediction1.utils.utils_ import *


def test_(args):
    device = args.device
    train_data, val_data, test_data, laplace, std_mean = load_data(args)
    model = Model(laplace, train_data.shape, args.prediction_time, device).to(device)
    PARENT_DIR = args.weights_file
    print(PARENT_DIR)
    print(PARENT_DIR, f"/{args.pems}_{args.prediction_time}.pth")
    model.load_state_dict(
        torch.load(os.path.join(PARENT_DIR
                                , f"{args.pems}_{args.prediction_time}.pth"),
                   map_location=device))
    print(PARENT_DIR, f"/{args.pems}_{args.prediction_time}.pth")

    test_loader = DataLoader(TrafficDataset1(test_data, std_mean, args.prediction_time),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)


    loss_mae_epoch = 0
    loss_mape_epoch = 0
    loss_rmse_epoch = 0
    test_len = 0
    model.eval()
    with torch.no_grad():
        for batch, (y, label) in enumerate(test_loader):
            y = y.to(device)
            label = label.to(device)
            pred = model(y, label[:, :, :, 1:])
            pred = pred * std_mean[0][0] + std_mean[0][1]
            loss_mae_batch = MAE_loss(pred, label[:, :, :, 0:1])
            loss_mape_batch = MAPE_loss(pred, label[:, :, :, 0:1])
            loss_rmse_batch = RMSE_loss(pred, label[:, :, :, 0:1])
            print(f'in batch {batch} mae loss: {loss_mae_batch:2f}, mape loss: {loss_mape_batch:2f}, rmse loss: {loss_rmse_batch:2f}')
            loss_mae_epoch += loss_mae_batch.item() * y.shape[0]
            loss_mape_epoch += loss_mape_batch.item() * y.shape[0]
            loss_rmse_epoch += loss_rmse_batch.item() * y.shape[0]
            test_len = test_len + y.shape[0]
            del y, label, pred, loss_mae_batch, loss_mape_batch, loss_rmse_batch
        loss_mae = loss_mae_epoch / test_len
        loss_mape = loss_mape_epoch / test_len
        loss_rmse = loss_rmse_epoch / test_len
        print(f"test average mpe loss: {loss_mae:.4f}%, mape loss: {loss_mape:.4f}%, rmse loss: {loss_rmse:.4f}%")

def main():
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
    PROJECT_ROOT= os.path.dirname(PARENT_DIR)  # 返回上一级目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--pems', type=str, default='PEMS04',
                        help='PEMS03, PEMS04, PEMS07, PEMS08')
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
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
    parser.add_argument('--weights_file', type=str,
                        default=os.path.join(PROJECT_ROOT, r'data/weights_file/xiaorong_1'))
    parser.add_argument('--device', type=str, default='cuda', help='CPU or cuda')
    parser.add_argument('--file_path', default=PROJECT_ROOT, help='file_path')  # 日志文件
    parser.add_argument('--wait', type=int, default=10)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_(args)


if __name__ == '__main__':
    main()
    print("-----")