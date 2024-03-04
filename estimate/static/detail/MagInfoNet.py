import torch
import torch.nn as nn
import sys
sys.path.append('..')
import func.process as pro
import func.net as net


class MagInfoNet(nn.Module):
    def __init__(self, gnn_style, adm_style, k, device):
        super(MagInfoNet, self).__init__()
        self.linear_at = nn.Sequential(nn.Linear(2, 1000), nn.Linear(1000, 6000))
        self.linear_t = nn.Sequential(nn.Linear(1, 1000), nn.Linear(1000, 6000))
        self.gnn_style = gnn_style
        self.ei1, self.ew1 = net.get_edge_info(k, 600, adm_style, device)
        self.ei2, self.ew2 = net.get_edge_info(k, 600, adm_style, device)
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1))
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.cnn4 = nn.Conv2d(32, 1, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
        self.bn1, self.bn2 = nn.BatchNorm2d(32), nn.BatchNorm2d(32)
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.linear2 = nn.Linear(3, 300)
        self.cnn5 = nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn10 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2))

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 10))
        self.gnn1 = net.get_gnn(gnn_style, 32, 32)
        self.gnn2 = net.get_gnn(gnn_style, 32, 1)
        self.last = nn.Linear(600, 1)

    def forward(self, x, ps_at, p_t):
        h_at = self.linear_at(ps_at).unsqueeze(1)
        h_t = self.linear_t(p_t).unsqueeze(1)

        h_x = self.cnn1(x.unsqueeze(1))
        h_x_0 = h_x
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn2(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn3(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = h_x + h_x_0
        h_x = self.cnn4(self.bn1(h_x))
        h_x = self.pool1(h_x)
        h_x = h_x.squeeze(1)

        out = torch.cat((h_x, h_at, h_t), dim=1)
        out = self.cnn5(out.unsqueeze(1))
        out_0 = out
        out = self.pre(self.bn2(out))
        out = self.cnn6(out)
        out = self.pre(self.bn2(out))
        out = self.cnn7(out)
        out = out + out_0

        out_1 = out
        out = self.cnn8(self.pre(self.bn2(out)))
        out = self.cnn9(self.pre(self.bn2(out)))
        out = out + out_1

        out_2 = out
        out = self.cnn10(self.pre(self.bn2(out)))
        out = self.cnn11(self.pre(self.bn2(out)))
        out = out + out_2

        out_3 = out
        out = self.cnn12(self.pre(self.bn2(out)))
        out = self.cnn13(self.pre(self.bn2(out)))
        out = out + out_3
        out = self.pool2(out)

        out = out.view(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
        put = net.run_gnn(self.gnn_style, self.gnn1, out, self.ei1, self.ew1)
        put = net.run_gnn(self.gnn_style, self.gnn2, put, self.ei2, self.ew2)
        put = self.last(put.view(put.shape[0], -1))
        return put.view(-1)


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

    ps_at_train, ps_at_test, p_t_train, p_t_test = pro.get_mai_data(df_train, df_test)

    train_dataset = pro.SelfData(data_train, sm_train, ps_at_train, p_t_train)
    test_dataset = pro.SelfData(data_test, sm_test, ps_at_test, p_t_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for x, y, ps_at, p_t, _ in test_loader:
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Shape of ps_at: {ps_at.shape}")
        print(f"Shape of p_t: {p_t.shape}")
        break


def code_train():
    def train():
        train_pred, train_true = [], []
        for item, (x, y, ps_at, p_t, _) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            ps_at, p_t = ps_at.to(device), p_t.to(device)

            optimizer.zero_grad()
            output = model(x, ps_at, p_t)
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
        for item, (x, y, ps_at, p_t, _) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            ps_at, p_t = ps_at.to(device), p_t.to(device)

            output = model(x, ps_at, p_t)
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

    model = MagInfoNet("unimp", "ts_un", 1, device).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_status = DlModelStatus.objects.get(name="MagInfoNet")

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
