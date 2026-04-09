import gc

from torch import optim, nn
from torch.amp import autocast, GradScaler

from traffic_prediction1.model.model3 import Model
from traffic_prediction1.utils.utils_ import *
from torch.utils.data import DataLoader
import time


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_(args):
    set_seed(42)
    filename = f"{args.pems}_{args.prediction_time}.pth"
    device = args.device
    print(f"use device: {device}")
    log_path = os.path.join(args.log_file, args.pems + str(args.prediction_time) + '.log')
    print("log_path:", log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # 自动创建缺失的文件路径
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'{args.pems}_{args.prediction_time}_交通预测模型\n')
        print(f"{log_path} 已创建。")
    train_data, val_data, test_data, laplace, std_mean = load_data(args)
    print(std_mean)
    print("*************************************************")
    train_total_loss = list()
    val_total_loss = list()
    val_loss_min = float('inf')  # 'inf'正无穷大
    best_model_wts = None
    wait = 0
    model_path = os.path.join(args.weights_file, filename)
    model = Model(laplace, train_data.shape, args.prediction_time, device).to(device)
    model = model.to(torch.float32)  # 强制保持 float32
    if os.path.exists(model_path) and args.pre_training:
        print("✅ 模型文件存在，启动预训练：", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型所在设备：", next(model.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    use_amp = args.device == "cuda"  # 如果你是用字符串传设备名
    scaler = GradScaler(device='cuda', enabled=use_amp)

    train_loader = DataLoader(TrafficDataset1(train_data, std_mean, args.prediction_time),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_len = train_data.shape[0] - 24 - int(args.prediction_time/5)
    val_loader = DataLoader(TrafficDataset1(val_data, std_mean, args.prediction_time),
                            batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_len = val_data.shape[0] - 24 - int(args.prediction_time/5)
    del train_data, val_data, test_data, laplace

    # 启用异常检测
    torch.autograd.set_detect_anomaly(False)

    for epoch in range(1, args.epochs + 1):
        if wait >= args.wait:
            log_write(log_path, f'early stop at epoch: {epoch:04d}')
            break
        loss_train = 0
        model.train()
        start_time = time.time()
        for batch, (x, label) in enumerate(train_loader):
            x = x.to(device) #torch.Size([8, 24, 307, 3])
            label = label.to(device) #torch.Size([8, 6, 307, 3])
            optimizer.zero_grad()  # 清楚所有梯度
            #device_ = torch.device(args.device)  # args.device 是字符串，如 'cuda' 或 'cpu'
            try:
                with autocast(device_type='cuda', dtype=torch.float32):
                    pred = model(x, label[:, :, :, 1:])
                    pred = pred * std_mean[0][0] + std_mean[0][1]
                    loss_batch = MAE_loss(pred, label[:, :, :, 0:1])
                scaler.scale(loss_batch).backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)  # 加这一行
                scaler.step(optimizer)
                scaler.update()
                loss_train += loss_batch.item() * (x.shape[0])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                patience = int(len(train_loader) / 10)
                if (batch + 1) % patience == 0:
                    print(
                        f'in epoch:{epoch} Training batch: {batch + 1}/{len(train_loader)} , training batch loss:{loss_batch:.4f}%')
                del x, label, pred, loss_batch
                torch.cuda.empty_cache()
                gc.collect()
            except RuntimeError as e:
                print(f"Error during backward pass at batch {batch} in epoch {epoch}: {e}")
                continue  # 或者可以选择停止训练

        loss_train = loss_train / train_len
        train_total_loss.append(loss_train)
        end_time = time.time()
        print(f"in epoch:{epoch} Training loss: {loss_train:.4f}%, epoch time:{end_time - start_time}")

        loss_val = 0
        model.eval()
        with torch.no_grad():
            for y, label in val_loader:
                y = y.to(device)
                label = label.to(device)
                pred = model(y, label[:, :, :, 1:])
                pred = pred * std_mean[0][0] + std_mean[0][1]
                loss_batch = MAE_loss(pred, label[:, :, :, 0:1])
                loss_val += loss_batch.item() * (y.shape[0])
                del y, label, pred, loss_batch
            loss_val = loss_val / val_len
            val_total_loss.append(loss_val)
            print(f"in epoch:{epoch} Validation loss: {loss_val:.4f}%")
            if loss_val < val_loss_min:
                wait = 0
                val_loss_min = loss_val
                best_model_wts = model.state_dict()
                log_write(log_path, f'val loss decrease from '
                                    f'{val_loss_min:.4f} to {loss_val:.4f}')
            else:
                wait += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Epoch {epoch}] 当前学习率: {optimizer.param_groups[0]['lr']}")
        torch.cuda.empty_cache()
        scheduler.step()
    model.load_state_dict(best_model_wts)

    # 确保目录存在
    os.makedirs(args.weights_file, exist_ok=True)

    # 格式化文件名
    save_path = os.path.join(args.weights_file, filename)

    # 推荐做法：只保存模型参数e
    torch.save(model.state_dict(), save_path)
    for epoch in range(len(train_total_loss)):
        print("__________")
        log_write(log_path, f"in epoch:{epoch} train loss: {train_total_loss[epoch]:.4f}%")
        log_write(log_path, f"in epoch:{epoch}  val  loss: {val_total_loss[epoch]:.4f}%")
    print(f"模型保存成功：{save_path}")

