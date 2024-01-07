import os

import pandas as pd
import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import func.net as net
import func.process as pro


class Net:
    def __init__(self):
        self.lr = 0.0005
        self.decay = 0.0005
        self.batch_size = 64
        self.epochs = 100

        self.root = None
        self.re_ad = None
        self.device = "cuda:1"
        self.m = 100
        self.name = "chunk2"
        self.sm_scale = ["ml"]
        self.m_train = 75
        self.m_test = 25
        self.network = "EQGraphNet"
        self.idx_train = None
        self.idx_test = None
        self.model = None

    def pre_train(self, input_data):
        input_data = pd.DataFrame(input_data, index=[0])
        self.lr = float(input_data["lr"].values[0])
        self.decay = float(input_data["decay"].values[0])
        self.batch_size = int(input_data["batch_size"].values[0])
        self.epochs = int(input_data["epochs"].values[0])

        self.sm_scale = input_data["sm_scale"].values[0]
        self.name = input_data["name"].values[0]
        self.device = input_data["device"].values[0]

        train_ratio = input_data["train_ratio"].values[0]
        self.m = int(input_data["m"].values[0])
        self.m_train = int(self.m * train_ratio)
        self.m_test = self.m - self.m_train

        self.network = input_data["network"].values[0]
        self.root = osp.join(input_data["root"].values[0], self.name)
        if self.network == "ConvNetQuakeINGV":
            self.network = "ConvNetQuake_INGV"
        elif self.network == "MagInfoNet":
            self.network = "MagInf"
        self.re_ad = osp.join(input_data["re_ad"].values[0], self.network)

        np.random.seed(100)
        self.idx_train, _ = pro.get_train_or_test_idx(self.m, self.m_train)
        return None

    def pre_test(self, input_data):
        input_data = pd.DataFrame(input_data, index=[0])
        self.sm_scale = input_data["sm_scale"].values[0]
        self.name = input_data["name"].values[0]
        self.device = input_data["device"].values[0]

        train_ratio = input_data["train_ratio"].values[0]
        self.m = int(input_data["m"].values[0])
        self.m_train = int(self.m * train_ratio)
        self.m_test = self.m - self.m_train

        self.network = input_data["network"].values[0]
        self.root = osp.join(input_data["root"].values[0], self.name)
        if self.network == "ConvNetQuakeINGV":
            self.network = "ConvNetQuake_INGV"
        elif self.network == "MagInfoNet":
            self.network = "MagInf"
        self.re_ad = osp.join(input_data["re_ad"].values[0], self.network)

        np.random.seed(100)
        _, self.idx_test = pro.get_train_or_test_idx(self.m, self.m_train)
        self.model.load_state_dict(
            torch.load(osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".
                                format(self.sm_scale, self.name, self.m_train, self.m_test))))
        self.model.to(self.device)
        return None


# generate dataset for model testing
def load_data(root, name, m, idx, device, sm_scale):
    data = torch.load(osp.join(root, str(m), "data.pt"))
    index = torch.load(osp.join(root, str(m), "index.pt"))
    df = pd.read_csv(osp.join(root, name + ".csv"))
    df = df.iloc[index, :]

    data = data[idx, :, :].float().to(device)
    df = df.iloc[idx, :]
    sm = torch.from_numpy(df["source_magnitude"].values.reshape(-1)).float()
    return pro.remain_sm_scale(data, df, sm, sm_scale)


# calculate result for response
def cal_metrics(true, pred, network, sm_scale, m):
    rmse = net.cal_rmse_one_arr(true, pred)
    r2 = net.cal_r2_one_arr(true, pred)
    num_show, num_round = 15, 2
    if network == "MagInf":
        network = "MagInfoNet"
    result = {
        'scale': sm_scale,
        'num':  m,
        'model': network,
        'rmse': np.round(float(rmse), 4),
        'r2': np.round(float(r2), 4),
        'pred': "  ".join('{:.{}f}'.format(one, num_round) for one in pred[:num_show]),
        'true': "  ".join('{:.{}f}'.format(one, num_round) for one in true[:num_show]),
    }
    return result


class MagInfoNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.MagInfoNet("unimp", "ts_un", 2, "cuda:1")

    def get_pt(self, df):
        ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
        _, ps_at = pro.prep_pt("sta", df.loc[:, ps_at_name].values)
        ps_at = torch.from_numpy(ps_at).float()

        t_name = ["p_travel_sec"]
        _, p_t = pro.prep_pt("sta", df.loc[:, t_name].values)
        p_t = torch.from_numpy(p_t).float()
        return ps_at, p_t

    def training(self, input_data):
        self.pre_train(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_train, self.device,
                                                   self.sm_scale)
        ps_at, p_t = self.get_pt(df)
        dataset = pro.SelfData(data, sm, ps_at, p_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true = [], []
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
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        # save model weights
        if not osp.exists(self.re_ad):
            os.makedirs(self.re_ad)
        torch.save(self.model.state_dict(),
                   osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, self.name, self.m_train, self.m_test)))

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

    def testing(self, input_data):
        self.pre_test(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_test, self.device,
                                                   self.sm_scale)
        ps_at, p_t = self.get_pt(df)
        dataset = pro.SelfData(data, sm, ps_at, p_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true = [], []
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

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)


class EQGraphNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.EQGraphNet("gcn", "ts_un", 1, "cuda:1")

    def training(self, input_data):
        self.pre_train(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_train, self.device,
                                                   self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true = [], []
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
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        # save model weights
        if not osp.exists(self.re_ad):
            os.makedirs(self.re_ad)
        torch.save(self.model.state_dict(),
                   osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, self.name, self.m_train, self.m_test)))

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

    def testing(self, input_data):
        self.pre_test(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_test, self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true = [], []
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

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)


class MagNet(Net):
    def __init__(self):
        super().__init__()
        self.model = net.MagNet()

    def training(self, input_data):
        self.pre_train(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_train, self.device,
                                                   self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true = [], []
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
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        # save model weights
        if not osp.exists(self.re_ad):
            os.makedirs(self.re_ad)
        torch.save(self.model.state_dict(),
                   osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, self.name, self.m_train, self.m_test)))

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

    def testing(self, input_data):
        self.pre_test(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_test, self.device,
                                                   self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true = [], []
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

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)


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

    def training(self, input_data):
        self.pre_train(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_train, self.device,
                                                   self.sm_scale)
        x, y = self.get_xy(data, df, sm, 125)
        dataset = pro.SelfData(x, y, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true = [], []
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
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        # save model weights
        if not osp.exists(self.re_ad):
            os.makedirs(self.re_ad)
        torch.save(self.model.state_dict(),
                   osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, self.name, self.m_train, self.m_test)))

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

    def testing(self, input_data):
        self.pre_test(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_test, self.device, self.sm_scale)

        x, y = self.get_xy(data, df, sm, 125)
        dataset = pro.SelfData(x, y, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true = [], []
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

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)


class ConvNetQuakeINGV(Net):
    def __init__(self):
        super().__init__()
        self.model = net.ConvNetQuakeINGV()

    def training(self, input_data):
        self.pre_train(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_train, self.device,
                                                   self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        self.model.to(self.device)

        pred, true = [], []
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
            print("Epoch: {:03d}  RMSE: {:.4f}  R2: {:.8f}".format(epoch, rmse, r2))

        # save model weights
        if not osp.exists(self.re_ad):
            os.makedirs(self.re_ad)
        torch.save(self.model.state_dict(),
                   osp.join(self.re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, self.name, self.m_train, self.m_test)))

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

    def testing(self, input_data):
        self.pre_test(input_data)
        data, sm, df, sm_scale, idx_sm = load_data(self.root, self.name, self.m, self.idx_test, self.device, self.sm_scale)
        dataset = pro.SelfData(data, sm)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pred, true = [], []
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

        return cal_metrics(true, pred, self.network, self.sm_scale, self.m)

