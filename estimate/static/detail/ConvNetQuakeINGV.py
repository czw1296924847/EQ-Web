import torch
import torch.nn as nn


class ConvNetQuakeINGV(nn.Module):
    def __init__(self):
        super(ConvNetQuakeINGV, self).__init__()
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv1d(3, 32, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.linear = nn.Linear(352, 127)

    def forward(self, x):
        h = self.cnn1(x)
        h = self.cnn2(self.relu(h))
        h = self.cnn3(self.relu(h))
        h = self.cnn4(self.relu(h))
        h = self.cnn5(self.relu(h))
        h = self.cnn6(self.relu(h))
        h = self.cnn7(self.relu(h))
        h = self.cnn8(self.relu(h))
        h = self.cnn9(self.relu(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        out = torch.mean(out[:, 52:72], dim=1)
        return out


def code_lib():
    import pandas as pd
    import torch
    import os.path as osp
    import numpy as np
    from torch import nn
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.nn import Parameter
    # from .models import DlModel, DlModelStatus
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

    train_dataset = pro.SelfData(data_train, sm_train)
    test_dataset = pro.SelfData(data_test, sm_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for x, y, _ in test_loader:
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")
        break


def code_train():
    def train():
        train_pred, train_true = [], []
        for item, (x, y, _) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_pred_one = output.detach().cpu().numpy()
            train_true_one = y.detach().cpu().numpy()
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
        for item, (x, y, _) in enumerate(tqdm(test_loader)):
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)

            test_pred_one = output.detach().cpu().numpy()
            test_true_one = y.detach().cpu().numpy()
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

    model = ConvNetQuakeINGV().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_status = DlModelStatus.objects.get(name="ConvNetQuakeINGV")

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
