import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def get_train_or_test_idx(num, num_train):
    idx_all = np.arange(num)
    idx_train = np.random.choice(num, num_train, replace=False)
    idx_test = np.array(list(set(idx_all) - set(idx_train)))
    return idx_train, idx_test


def be_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x
    else:
        raise TypeError("x must be tensor or ndarray, but gut {}".format(type(x)))


def get_item_by_dim(data, item):
    if torch.is_tensor(data):
        n_dim = data.dim()
    elif type(data) == np.ndarray:
        n_dim = data.ndim
    else:
        raise TypeError("The input must be torch.tensor or numpy.ndarray!")
    if n_dim == 1:
        return data[item]
    elif n_dim == 2:
        return data[item, :]
    elif n_dim == 3:
        return data[item, :, :]
    elif n_dim == 4:
        return data[item, :, :, :]
    else:
        raise ValueError("Unknown dim() of input!")


class SelfData(Dataset):
    def __init__(self, data, label, *args):
        super(SelfData, self).__init__()
        self.data = be_tensor(data)
        self.label = be_tensor(label)
        self.args = args
        self.data_else = self.get_data_else()

    def get_data_else(self):
        num = len(self.args)
        data_else = [0] * num
        if num != 0:
            for i in range(num):
                data_else_one = self.args[i]
                data_else[i] = data_else_one
        return data_else

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        data_one = get_item_by_dim(self.data, item)
        label_one = get_item_by_dim(self.label, item)
        result = [data_one, label_one]
        if len(self.data_else) != 0:
            num = len(self.data_else)
            data_else_one = [0] * num
            for i in range(num):
                x = self.data_else[i]
                x_one = get_item_by_dim(x, item)
                data_else_one[i] = x_one
            result = result + data_else_one
        result.append(item)
        return tuple(result)


def remain_sm_scale(data, df, label, scale):
    if isinstance(scale, list):
        smt = df['source_magnitude_type'].isin(scale).values
        idx_sm = np.argwhere(smt).reshape(-1)
        num = len(scale)
        scale_name = ""
        for i in range(num):
            if i == 0:
                scale_name = scale[i]
            else:
                scale_name = scale_name + "_" + scale[i]
    else:
        smt = df.source_magnitude_type.values.reshape(-1)
        idx_sm = np.argwhere(smt == scale).reshape(-1)
        scale_name = scale
    data = data[idx_sm, :, :]
    label = label[idx_sm]
    df = df.iloc[idx_sm, :]
    return data, label, df, scale_name, idx_sm


def prep_pt(prep_style, train, test=None):
    if prep_style == "sta":
        model = StandardScaler()
    else:
        raise TypeError("Unknown Type of prep_style!")
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    model.fit(train)
    train_prep = model.transform(train)
    if test is None:
        return model, train_prep
    if test.ndim == 1:
        test = test.reshape(-1, 1)
    test_prep = model.transform(test)
    return model, train_prep, test_prep
