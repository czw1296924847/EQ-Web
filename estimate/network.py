import os

import pandas as pd
import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import Parameter
import func.net as net
import func.process as pro


class Net:
    def __init__(self):
        self.lr = 0.0005
        self.decay = 0.0005
        self.batch_size = 64
        self.epochs = 100

        self.root = pro.ROOT
        self.re_ad = pro.RE_AD
        self.device = "cpu"
        self.data_size = 100
        self.chunk_name = "chunk2"
        self.sm_scale = ["ml"]
        self.data_size_train = 75
        self.data_size_test = 25
        self.model_name = "EQGraphNet"
        self.idx_train = None
        self.idx_test = None
        self.model = None

    def pre_train(self, input_data, model_name):
        input_data = pd.DataFrame(input_data, index=[0])
        self.lr = float(input_data["lr"].values[0])
        self.batch_size = int(input_data["batch_size"].values[0])
        self.epochs = int(input_data["epochs"].values[0])

        self.sm_scale = input_data["sm_scale"].values[0]
        self.chunk_name = input_data["chunk_name"].values[0]
        self.device = input_data["device"].values[0]

        train_ratio = float(input_data["train_ratio"].values[0])
        self.data_size = int(input_data["data_size"].values[0])
        self.data_size_train = int(self.data_size * train_ratio)
        self.data_size_test = self.data_size - self.data_size_train

        self.model_name = model_name
        # self.root = osp.join(input_data["root"].values[0], self.chunk_name)
        # self.re_ad = osp.join(input_data["re_ad"].values[0], self.model_name)
        if self.root.split('/')[-1] != self.chunk_name:
            self.root = osp.join(self.root, self.chunk_name)
        if self.re_ad.split('/')[-1] != self.model_name:
            self.re_ad = osp.join(self.re_ad, self.model_name)

        np.random.seed(100)
        self.idx_train, _ = pro.get_train_or_test_idx(self.data_size, self.data_size_train)
        return None

    def pre_test(self, input_data, model_name):
        input_data = pd.DataFrame(input_data, index=[0])
        self.sm_scale = input_data["sm_scale"].values[0]
        self.chunk_name = input_data["chunk_name"].values[0]
        self.device = input_data["device"].values[0]

        train_ratio = float(input_data["train_ratio"].values[0])
        self.data_size = int(input_data["data_size"].values[0])
        self.data_size_train = int(self.data_size * train_ratio)
        self.data_size_test = self.data_size - self.data_size_train

        self.model_name = model_name
        # self.root = osp.join(input_data["root"].values[0], self.chunk_name)
        # self.re_ad = osp.join(input_data["re_ad"].values[0], self.model_name)
        if self.root.split('/')[-1] != self.chunk_name:
            self.root = osp.join(self.root, self.chunk_name)
        if self.re_ad.split('/')[-1] != self.model_name:
            self.re_ad = osp.join(self.re_ad, self.model_name)

        np.random.seed(100)
        _, self.idx_test = pro.get_train_or_test_idx(self.data_size, self.data_size_train)
        self.model.load_state_dict(
            torch.load(osp.join(self.re_ad, str(self.data_size), "model_{}_{}_{}_{}.pkl".
                                format(self.sm_scale, self.chunk_name, self.data_size_train, self.data_size_test))))
        self.model.to(self.device)
        return None


# generate dataset for model testing
def load_data(root, chunk_name, data_size, idx, device, sm_scale):
    data = torch.load(osp.join(root, str(data_size), "data.pt"))
    index = torch.load(osp.join(root, str(data_size), "index.pt"))
    df = pd.read_csv(osp.join(root, chunk_name + ".csv"))
    df = df.iloc[index, :]

    data = data[idx, :, :].float().to(device)
    df = df.iloc[idx, :]
    sm = torch.from_numpy(df["source_magnitude"].values.reshape(-1)).float()
    return pro.remain_sm_scale(data, df, sm, sm_scale)


# calculate result for response
def get_metrics(true, pred, model_name, sm_scale, data_size):
    r2, rmse, e_mean, e_std = net.cal_metrics(true, pred)
    num_show, num_round = 15, 2
    result = {
        'sm_scale': sm_scale,
        'data_size': data_size,
        'model_name': model_name,
        'rmse': str(np.round(float(rmse), 4)),
        'r2': str(np.round(float(r2), 4)),
        'e_mean': str(np.round(float(e_mean), 4)),
        'e_std': str(np.round(float(e_std), 4)),
        'pred': "  ".join('{:.{}f}'.format(one, num_round) for one in pred[:num_show]),
        'true': "  ".join('{:.{}f}'.format(one, num_round) for one in true[:num_show]),
    }
    print(result)
    return result


def ei_ew_device(model_name, model, device):
    if model_name == "MagInfoNet":
        model.ei1, model.ew1 = model.ei1.to(device), Parameter(model.ew1.float().to(device))
        model.ei2, model.ew2 = model.ei2.to(device), Parameter(model.ew2.float().to(device))
    elif model_name == "EQGraphNet":
        model.ei1, model.ew1 = model.ei1.to(device), Parameter(model.ew1.float().to(device))
        model.ei2, model.ew2 = model.ei2.to(device), Parameter(model.ew2.float().to(device))
        model.ei3, model.ew3 = model.ei3.to(device), Parameter(model.ew3.float().to(device))
        model.ei4, model.ew4 = model.ei4.to(device), Parameter(model.ew4.float().to(device))
        model.ei5, model.ew5 = model.ei5.to(device), Parameter(model.ew5.float().to(device))
        model.ei6, model.ew6 = model.ei6.to(device), Parameter(model.ew6.float().to(device))
        model.ei7, model.ew7 = model.ei7.to(device), Parameter(model.ew7.float().to(device))
        model.ei8, model.ew8 = model.ei8.to(device), Parameter(model.ew8.float().to(device))
        model.ei9, model.ew9 = model.ei9.to(device), Parameter(model.ew9.float().to(device))
        model.ei10, model.ew10 = model.ei10.to(device), Parameter(model.ew10.float().to(device))
    else:
        raise TypeError("Unknown type of 'model_name")
    return model


class MagInfoNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.MagInfoNet("unimp", "ts_un", 1, "cpu")

    def get_pt(self, df):
        ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
        _, ps_at = pro.prep_pt("sta", df.loc[:, ps_at_name].values)
        ps_at = torch.from_numpy(ps_at).float()

        t_name = ["p_travel_sec"]
        _, p_t = pro.prep_pt("sta", df.loc[:, t_name].values)
        p_t = torch.from_numpy(p_t).float()
        return ps_at, p_t

    def training(self, input_data, model_name):
        self.pre_train(input_data, model_name)
        self.model = ei_ew_device(self.model_name, self.model, self.device)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_train,
                                                   self.device, self.sm_scale)
        ps_at, p_t = self.get_pt(df)
        dataset = pro.SelfData(data, sm, ps_at, p_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + " Start {} Training ".format(self.model_name) + "=" * 20 + "\n")
        for epoch in range(self.epochs):
            for item, (x, y, ps_at, p_t, _) in enumerate(tqdm(loader)):
                x, y = x.to(self.device), y.to(self.device)
                ps_at, p_t = ps_at.to(self.device), p_t.to(self.device)

                optimizer.zero_grad()
                output = self.model(x, ps_at, p_t)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                pred_one = output.detach().cpu().numpy()
                true_one = y.detach().cpu().numpy()
                if item == 0:
                    pred = pred_one
                    true = true_one
                else:
                    pred = np.concatenate((pred, pred_one), axis=0)
                    true = np.concatenate((true, true_one), axis=0)
            rmse = net.cal_rmse_one_arr(true, pred)
            r2 = net.cal_r2_one_arr(true, pred)
            loss_curve.append((rmse ** 2))
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        pro.save_result("train", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test, self.model)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)

    def testing(self, input_data, model_name):
        self.pre_test(input_data, model_name)
        self.model = ei_ew_device(self.model_name, self.model, self.device)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_test,
                                                   self.device, self.sm_scale)
        ps_at, p_t = self.get_pt(df)
        dataset = pro.SelfData(data, sm, ps_at, p_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + "Start {} Testing".format(self.model_name) + "=" * 20 + "\n")
        for item, (x, y, ps_at, p_t, _) in enumerate(tqdm(loader)):
            x, y = x.to(self.device), y.to(self.device)
            ps_at, p_t = ps_at.to(self.device), p_t.to(self.device)

            output = self.model(x, ps_at, p_t)
            pred_one = output.detach().cpu().numpy()
            true_one = y.detach().cpu().numpy()
            if item == 0:
                pred = pred_one
                true = true_one
            else:
                pred = np.concatenate((pred, pred_one), axis=0)
                true = np.concatenate((true, true_one), axis=0)
        rmse = net.cal_rmse_one_arr(true, pred)
        r2 = net.cal_r2_one_arr(true, pred)
        loss_curve.append((rmse ** 2))
        print("RMSE: {:.4f}  R2: {:.8f}".format(rmse, r2))

        pro.save_result("test", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)


class EQGraphNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.EQGraphNet("gcn", "ts_un", 1, "cpu")

    def training(self, input_data, model_name):
        self.pre_train(input_data, model_name)
        self.model = ei_ew_device(self.model_name, self.model, self.device)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_train,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + " Start {} Training ".format(self.model_name) + "=" * 20 + "\n")
        for epoch in range(self.epochs):
            for item, (x, y, _) in enumerate(tqdm(loader)):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                pred_one = output.detach().cpu().numpy()
                true_one = y.detach().cpu().numpy()
                if item == 0:
                    pred = pred_one
                    true = true_one
                else:
                    pred = np.concatenate((pred, pred_one), axis=0)
                    true = np.concatenate((true, true_one), axis=0)
            rmse = net.cal_rmse_one_arr(true, pred)
            r2 = net.cal_r2_one_arr(true, pred)
            loss_curve.append((rmse ** 2))
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        pro.save_result("train", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test, self.model)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)

    def testing(self, input_data, model_name):
        self.pre_test(input_data, model_name)
        self.model = ei_ew_device(self.model_name, self.model, self.device)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_test,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + "Start {} Testing".format(self.model_name) + "=" * 20 + "\n")
        for item, (x, y, _) in enumerate(tqdm(loader)):
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            pred_one = output.detach().cpu().numpy()
            true_one = y.detach().cpu().numpy()
            if item == 0:
                pred = pred_one
                true = true_one
            else:
                pred = np.concatenate((pred, pred_one), axis=0)
                true = np.concatenate((true, true_one), axis=0)
        rmse = net.cal_rmse_one_arr(true, pred)
        r2 = net.cal_r2_one_arr(true, pred)
        loss_curve.append((rmse ** 2))
        print("RMSE: {:.4f}  R2: {:.8f}".format(rmse, r2))

        pro.save_result("test", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)


class MagNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.MagNet()

    def training(self, input_data, model_name):
        self.pre_train(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_train,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + " Start {} Training ".format(self.model_name) + "=" * 20 + "\n")
        for epoch in range(self.epochs):
            for item, (x, y, _) in enumerate(tqdm(loader)):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                pred_one = output.detach().cpu().numpy()
                true_one = y.detach().cpu().numpy()
                if item == 0:
                    pred = pred_one
                    true = true_one
                else:
                    pred = np.concatenate((pred, pred_one), axis=0)
                    true = np.concatenate((true, true_one), axis=0)
            rmse = net.cal_rmse_one_arr(true, pred)
            r2 = net.cal_r2_one_arr(true, pred)
            loss_curve.append((rmse ** 2))
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        pro.save_result("train", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test, self.model)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)

    def testing(self, input_data, model_name):
        self.pre_test(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_test,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + "Start {} Testing".format(self.model_name) + "=" * 20 + "\n")
        for item, (x, y, _) in enumerate(tqdm(loader)):
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            pred_one = output.detach().cpu().numpy()
            true_one = y.detach().cpu().numpy()
            if item == 0:
                pred = pred_one
                true = true_one
            else:
                pred = np.concatenate((pred, pred_one), axis=0)
                true = np.concatenate((true, true_one), axis=0)
        rmse = net.cal_rmse_one_arr(true, pred)
        r2 = net.cal_r2_one_arr(true, pred)
        loss_curve.append((rmse ** 2))
        print("RMSE: {:.4f}  R2: {:.8f}".format(rmse, r2))

        pro.save_result("test", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)


class CREIME(Net):
    def __init__(self):
        super().__init__()
        self.model = net.CREIME()

    def cal_mag(self, output):
        output_last = output[:, -10:]
        mag = torch.mean(output_last, dim=1)
        return mag

    def get_xy(self, data, df, sm, p_len):
        data, sm = data.detach().cpu().numpy(), sm.detach().cpu().numpy()
        num = data.shape[0]
        p_as = df.loc[:, "p_arrival_sample"].values.reshape(-1).astype(int)
        n_len = 512 - p_len
        y_n_i = np.ones(shape=(1, n_len)) * (-4)
        x, y = np.zeros(shape=(num, 3, 512)), np.zeros(shape=(num, 512))
        for i in range(num):
            p_as_i, sm_i = p_as[i], sm[i]
            if p_as_i > n_len:
                x_i = data[i, :, (p_as_i - n_len): (p_as_i + p_len)]
                y_i = np.hstack([y_n_i, np.ones(shape=(1, p_len)) * sm_i])
            else:
                x_i = data[i, :, :512]
                y_i = np.hstack([np.ones(shape=(1, p_as_i)) * (-4), np.ones(shape=(1, 512 - p_as_i)) * sm_i])

            x[i, :, :] = x_i
            y[i, :] = y_i
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        return x, y

    def training(self, input_data, model_name):
        self.pre_train(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_train,
                                                   self.device, self.sm_scale)
        x, y = self.get_xy(data, df, sm, 125)
        dataset = pro.SelfData(x, y, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + " Start {} Training ".format(self.model_name) + "=" * 20 + "\n")
        for epoch in range(self.epochs):
            for item, (x, y, sm, _) in enumerate(tqdm(loader)):
                x, y, sm = x.to(self.device), y.to(self.device), sm.to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                pred_one = self.cal_mag(output).detach().cpu().numpy()
                true_one = sm.detach().cpu().numpy()
                if item == 0:
                    pred = pred_one
                    true = true_one
                else:
                    pred = np.concatenate((pred, pred_one), axis=0)
                    true = np.concatenate((true, true_one), axis=0)
            rmse = net.cal_rmse_one_arr(true, pred)
            r2 = net.cal_r2_one_arr(true, pred)
            loss_curve.append((rmse ** 2))
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        pro.save_result("train", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test, self.model)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)

    def testing(self, input_data, model_name):
        self.pre_test(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_test,
                                                   self.device, self.sm_scale)

        x, y = self.get_xy(data, df, sm, 125)
        dataset = pro.SelfData(x, y, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + "Start {} Testing".format(self.model_name) + "=" * 20 + "\n")
        for item, (x, y, sm, _) in enumerate(tqdm(loader)):
            x, y, sm = x.to(self.device), y.to(self.device), sm.to(self.device)

            output = self.model(x)
            pred_one = self.cal_mag(output).detach().cpu().numpy()
            true_one = sm.detach().cpu().numpy()
            if item == 0:
                pred = pred_one
                true = true_one
            else:
                pred = np.concatenate((pred, pred_one), axis=0)
                true = np.concatenate((true, true_one), axis=0)
        rmse = net.cal_rmse_one_arr(true, pred)
        r2 = net.cal_r2_one_arr(true, pred)
        loss_curve.append((rmse ** 2))
        print("RMSE: {:.4f}  R2: {:.8f}".format(rmse, r2))

        pro.save_result("test", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)


class ConvNetQuakeINGV(Net):
    def __init__(self):
        super().__init__()
        self.model = net.ConvNetQuakeINGV()

    def training(self, input_data, model_name):
        self.pre_train(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_train,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + " Start {} Training ".format(self.model_name) + "=" * 20 + "\n")
        for epoch in range(self.epochs):
            for item, (x, y, _) in enumerate(tqdm(loader)):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                pred_one = output.detach().cpu().numpy()
                true_one = y.detach().cpu().numpy()
                if item == 0:
                    pred = pred_one
                    true = true_one
                else:
                    pred = np.concatenate((pred, pred_one), axis=0)
                    true = np.concatenate((true, true_one), axis=0)
            rmse = net.cal_rmse_one_arr(true, pred)
            r2 = net.cal_r2_one_arr(true, pred)
            loss_curve.append((rmse ** 2))
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        pro.save_result("train", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test, self.model)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)

    def testing(self, input_data, model_name):
        self.pre_test(input_data, model_name)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.chunk_name, self.data_size, self.idx_test,
                                                   self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true, loss_curve = [], [], []
        print("\n\n" + "=" * 20 + "Start {} Testing".format(self.model_name) + "=" * 20 + "\n")
        for item, (x, y, _) in enumerate(tqdm(loader)):
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            pred_one = output.detach().cpu().numpy()
            true_one = y.detach().cpu().numpy()
            if item == 0:
                pred = pred_one
                true = true_one
            else:
                pred = np.concatenate((pred, pred_one), axis=0)
                true = np.concatenate((true, true_one), axis=0)
        rmse = net.cal_rmse_one_arr(true, pred)
        r2 = net.cal_r2_one_arr(true, pred)
        loss_curve.append((rmse ** 2))
        print("RMSE: {:.4f}  R2: {:.8f}".format(rmse, r2))

        pro.save_result("test", osp.join(self.re_ad, str(self.data_size)), true, pred, loss_curve, sm_scale,
                        self.chunk_name, self.data_size_train, self.data_size_test)

        return get_metrics(true, pred, self.model_name, self.sm_scale, self.data_size)
