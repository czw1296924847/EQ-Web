import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# from https://doi.org/10.1029/2022JB024595
class CREIME(nn.Module):
    def __init__(self):
        super(CREIME, self).__init__()
        self.cnn1 = nn.Conv1d(3, 32, kernel_size=15, stride=1, padding=7)
        self.cnn2 = nn.Conv1d(32, 16, kernel_size=15, stride=1, padding=7)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=15, stride=1, padding=7)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.lstm1 = nn.LSTM(8, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.linear = nn.Linear(2048, 512)

    def forward(self, x):
        h = self.cnn1(x)
        h = self.pool(h)
        h = self.cnn2(h)
        h = self.pool(h)
        h = self.cnn3(h)
        h = self.pool(h)

        out, (_, _) = self.lstm1(h)
        out, (_, _) = self.lstm2(out)

        put = out.reshape(out.shape[0], -1)
        put = self.linear(put)
        return put


def code_lib():
    import pandas as pd
    import torch
    import os.path as osp
    import numpy as np
    from torch import nn
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.nn import Parameter
    from django.db import transaction
    import sys
    sys.path.append("..")
    import func.net as net
    import func.process as pro

    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
    import django
    django.setup()
    from estimate.models import DlModel, DlModelStatus


def code_data():
    def cal_mag(output):
        output_last = output[:, -10:]
        mag = torch.mean(output_last, dim=1)
        return mag

    def get_xy(data, df, sm_, p_len):
        data, sm_ = data.numpy(), sm_.numpy()
        num = data.shape[0]
        p_as = df["p_arrival_sample"].values.reshape(-1).astype(int)
        n_len = 512 - p_len
        y_n_i = np.ones(shape=(1, n_len)) * (-4)
        x_, y_ = np.zeros(shape=(num, 3, 512)), np.zeros(shape=(num, 512))
        for i in range(num):
            p_as_i, sm_i = p_as[i], sm_[i]
            if p_as_i > n_len:
                x_i = data[i, :, (p_as_i - n_len): (p_as_i + p_len)]
                y_i = np.hstack([y_n_i, np.ones(shape=(1, p_len)) * sm_i])
            else:
                x_i = data[i, :, :512]
                y_i = np.hstack([np.ones(shape=(1, p_as_i)) * (-4), np.ones(shape=(1, 512 - p_as_i)) * sm_i])

            x_[i, :, :] = x_i
            y_[i, :] = y_i
        x_, y_ = torch.from_numpy(x_).float(), torch.from_numpy(y_).float()
        return x_, y_

    chunk_name = "chunk2"
    root = "/home/chenziwei2021/standford_dataset/{}".format(chunk_name)
    sm_scale = ["md"]
    np.random.seed(100)
    batch_size = 64

    data_size = 200000
    train_ratio = 0.75
    data_size_train = int(data_size * train_ratio)
    data_size_test = data_size - data_size_train

    idx_train, idx_test = pro.get_train_or_test_idx(data_size, data_size_train)
    eq_train = pro.Chunk(data_size, True, data_size_train, idx_train, root, chunk_name)
    eq_test = pro.Chunk(data_size, False, data_size_train, idx_test, root, chunk_name)
    df_train, df_test = eq_train.df, eq_test.df

    data_train, data_test = eq_train.data.float(), eq_test.data.float()
    sm_train = torch.from_numpy(df_train["source_magnitude"].values.reshape(-1)).float()
    sm_test = torch.from_numpy(df_test["source_magnitude"].values.reshape(-1)).float()

    # Select samples according to Magnitude Type
    data_train, sm_train, df_train, _, _ = pro.remain_sm_scale(data_train, df_train, sm_train, sm_scale)
    data_test, sm_test, df_test, sm_scale, _ = pro.remain_sm_scale(data_test, df_test, sm_test, sm_scale)
    x_train, y_train = get_xy(data_train, df_train, sm_train, 125)
    x_test, y_test = get_xy(data_test, df_test, sm_test, 125)

    train_dataset = pro.SelfData(x_train, y_train, sm_train)
    test_dataset = pro.SelfData(x_test, y_test, sm_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for x, y, sm, _ in test_loader:
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Shape of sm: {sm.shape}")
        break


def code_train():
    def train():
        train_pred, train_true = [], []
        for item, (x, y, sm, _) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            sm = sm.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_pred_one = cal_mag(output).detach().cpu().numpy()
            train_true_one = sm.detach().cpu().numpy()
            if item == 0:
                train_pred = train_pred_one
                train_true = train_true_one
            else:
                train_pred = np.concatenate((train_pred, train_pred_one), axis=0)
                train_true = np.concatenate((train_true, train_true_one), axis=0)
        rmse_train = net.cal_rmse_one_arr(train_true, train_pred)
        r2_train = net.cal_r2_one_arr(train_true, train_pred)
        return rmse_train, r2_train


def code_test():
    def test():
        test_pred, test_true = [], []
        for item, (x, y, sm, _) in enumerate(tqdm(test_loader)):
            x, y = x.to(device), y.to(device)
            sm = sm.to(device)

            output = model(x)
            loss = criterion(output, y)

            test_pred_one = cal_mag(output).detach().cpu().numpy()
            test_true_one = sm.detach().cpu().numpy()
            if item == 0:
                test_pred = test_pred_one
                test_true = test_true_one
            else:
                test_pred = np.concatenate((test_pred, test_pred_one), axis=0)
                test_true = np.concatenate((test_true, test_true_one), axis=0)
        rmse_test = net.cal_rmse_one_arr(test_true, test_pred)
        r2_test = net.cal_r2_one_arr(test_true, test_pred)
        return rmse_test, r2_test


def code_run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr = 0.0005
    weight_decay = 0.0005
    epochs = 4

    model = CREIME().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_status = DlModelStatus.objects.get(name="CREIME")

    for epoch in range(epochs):
        rmse_train, r2_train = train()
        rmse_test, r2_test = test()

        """
        The 'process' during epochs, must follow the format of

        "Epoch:{epoch},{metric1}:{value1},{metric2}:{value2},...", such as

        "Epoch:0001,RMSE_Train:0.8888,RMSE_Test:0.7777,R2_Train:0.6666,R2_Test:0.5555"
        """
        process = "Epoch:{:04d},RMSE_Train:{:.4f},RMSE_Test:{:.4f},R2_Train:{:.8f},R2_Test:{:.8f}".format(
            epoch, rmse_train, rmse_test, r2_train, r2_test)
        model_status.process = process
        model_status.save()
        print(process)

    """
    The 'process' after epochs, must be added '#Done'
    """
    process = "#Done"
    model_status.process += process
    model_status.save()
    print(process)
