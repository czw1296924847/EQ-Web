import torch
import torch.nn as nn
import sys
sys.path.append('..')
import func.process as pro
import func.net as net


class EQGraphNet(nn.Module):
    def __init__(self, gnn_style, adm_style, k, device):
        super(EQGraphNet, self).__init__()
        self.relu = nn.ReLU()
        self.gnn_style = gnn_style
        self.adm_style = adm_style
        self.k, self.device = k, device
        self.pre = nn.Sequential(nn.ReLU())
        self.cnn1 = nn.Conv1d(3, 16, kernel_size=2, stride=2)
        self.cnn2 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn3 = nn.Conv1d(16, 16, kernel_size=2, stride=2)
        self.cnn4 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.cnn5 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn6 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.cnn7 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.cnn8 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn9 = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.cnn10 = nn.Conv1d(64, 128, kernel_size=2, stride=2)
        self.cnn11 = nn.Conv1d(128, 128, kernel_size=2, stride=2)
        self.linear = nn.Linear(256, 1)
        self.ei1, self.ew1 = net.get_edge_info(k, 3000, adm_style, device)
        self.ei2, self.ew2 = net.get_edge_info(k, 1500, adm_style, device)
        self.ei3, self.ew3 = net.get_edge_info(k, 750, adm_style, device)
        self.ei4, self.ew4 = net.get_edge_info(k, 375, adm_style, device)
        self.ei5, self.ew5 = net.get_edge_info(k, 187, adm_style, device)
        self.ei6, self.ew6 = net.get_edge_info(k, 93, adm_style, device)
        self.ei7, self.ew7 = net.get_edge_info(k, 46, adm_style, device)
        self.ei8, self.ew8 = net.get_edge_info(k, 23, adm_style, device)
        self.ei9, self.ew9 = net.get_edge_info(k, 11, adm_style, device)
        self.ei10, self.ew10 = net.get_edge_info(k, 5, adm_style, device)
        self.gnn1 = net.get_gnn(gnn_style, 16, 16)
        self.gnn2 = net.get_gnn(gnn_style, 16, 16)
        self.gnn3 = net.get_gnn(gnn_style, 16, 16)
        self.gnn4 = net.get_gnn(gnn_style, 32, 32)
        self.gnn5 = net.get_gnn(gnn_style, 32, 32)
        self.gnn6 = net.get_gnn(gnn_style, 32, 32)
        self.gnn7 = net.get_gnn(gnn_style, 64, 64)
        self.gnn8 = net.get_gnn(gnn_style, 64, 64)
        self.gnn9 = net.get_gnn(gnn_style, 64, 64)
        self.gnn10 = net.get_gnn(gnn_style, 128, 128)

    def forward(self, x):
        h_0 = h = self.cnn1(x)
        h = net.run_gnn(self.gnn_style, self.gnn1, h, self.ei1, self.ew1)
        h = h + h_0
        h_1 = h = self.cnn2(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn2, h, self.ei2, self.ew2)
        h = h + h_1
        h_2 = h = self.cnn3(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn3, h, self.ei3, self.ew3)
        h = h + h_2
        h_3 = h = self.cnn4(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn4, h, self.ei4, self.ew4)
        h = h + h_3
        h_4 = h = self.cnn5(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn5, h, self.ei5, self.ew5)
        h = h + h_4
        h_5 = h = self.cnn6(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn6, h, self.ei6, self.ew6)
        h = h + h_5
        h_6 = h = self.cnn7(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn7, h, self.ei7, self.ew7)
        h = h + h_6
        h_7 = h = self.cnn8(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn8, h, self.ei8, self.ew8)
        h = h + h_7
        h_8 = h = self.cnn9(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn9, h, self.ei9, self.ew9)
        h = h + h_8
        h_9 = h = self.cnn10(self.pre(h))
        h = net.run_gnn(self.gnn_style, self.gnn10, h, self.ei10, self.ew10)
        h = h + h_9
        h = self.cnn11(self.pre(h))

        out = h.view(h.shape[0], -1)
        out = self.linear(out)
        return out.view(-1)


def code_lib():
    import numpy as np
    import torch
    import torch.nn as nn
    import pandas as pd
    import os
    import os.path as osp
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import sys
    sys.path.append('..')
    import func.process as pro
    import func.net as net

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

    model = EQGraphNet("gcn", "ts_un", 1, device).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_status = DlModelStatus.objects.get(name="EQGraphNet")

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
