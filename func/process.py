import re
import pandas as pd
import h5py
import torch
import numpy as np
import os
import os.path as osp
import inspect
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

ROOT = "/home/chenziwei2021/standford_dataset"
RE_AD = "/home/chenziwei2021/pyn/paper/EQGraphNet/web/estimate/static/result"
DATA_AD = "/home/chenziwei2021/pyn/paper/EQGraphNet/web/estimate/static/data"
WORD_AD = "/home/chenziwei2021/pyn/paper/EQGraphNet/web"
PY_AD = "/home/chenziwei2021/anaconda3/envs/cmh/bin/python3.11"
DEFAULT_MODELS = ["MagInfoNet", "EQGraphNet", "MagNet", "CREIME", "ConvNetQuakeINGV"]
DEFAULT_LIBS = "import numpy as np\n"


def is_error(string, no_warn=True):
    """
    Check whether the output contents contain error messages

    :param string:
    :param no_warn: Ignore Warning from STEAD
    :return:
    """
    lines = string.split('\n')
    if no_warn:
        pattern = re.compile(r'.*DtypeWarning.*|.*self\.df = pd\.read_csv.*', re.MULTILINE)
        lines = [line for line in lines if not pattern.search(line) and line != '']
    errors = ['error', 'Error']
    return any(error in str(line) for error in errors for line in lines)


def get_source(obj, no_def=True):
    """
    Transform code to string

    :param obj: Can be a module, class, method, function, traceback, frame, or code object
    :param no_def: Remove definition and corresponding indents
    :return:
    """
    source = inspect.getsource(obj)
    if no_def:
        # replace the definition statement with the empty string by re
        no_def_source = re.sub(r'^\s*def\s+code_\w+\s*\(.*\):(?!.*\n\s*def\s+code_\w+\s*\(.*\):)', '', source, flags=re.MULTILINE)

        # find and remove the minimum indentation for non-empty lines
        lines = no_def_source.split('\n')
        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.lstrip())
        lines = lines[1:-1]
        source = '\n'.join(line[min_indent:] for line in lines)
    return source


def get_lib_by_files(files, remain=True):
    """
    Get dependent libraries like "import numpy as np\nimport torch", from files

    :param files: List of file names, like ["network.py", "process.py"]
    :param remain: Remain (True) or remove (False) based on regular expression
    :return:
    """
    codes = []
    for file in files:
        with open(file, 'r') as f:
            contents = f.read()
            code = get_library_by_string(contents, remain)
            codes += code
    library = "\n".join(codes)

    # add default library in the beginning (must be added! Or you can try removing what might happen)
    library = DEFAULT_LIBS + library
    return library


def get_library_by_string(string, remain):
    """
    Get dependent libraries , from string

    :param string: like "import numpy as np\nimport torch"
    :param remain:
    :return: like ["import numpy as np", "import torch"]
    """
    import_pattern = re.compile(
        r"^(from\s+[\w.]+\s+import\s+([\w.]+(\s*,\s*[\w.]+)*)(\s+as\s+\w+)?)|(^import\s+([\w.]+(\s*,\s*[\w.]+)*)(\s+as\s+\w+)?)|^(sys\.path\.append\(\"..\"\))",
        re.MULTILINE)
    if remain:
        contents = import_pattern.findall(string)
        code = [max(content, key=len) for content in contents]
    else:
        code = [content for content in string.splitlines() if not import_pattern.match(content)]
    return code


def duplicate_lib(string):
    code = [content for content in string.splitlines()]
    return "\n".join(list(dict.fromkeys(code)))


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


def save_result(style, re_ad, true, pred, loss, sm_scale, name, m_train, m_test, model=None):
    if not osp.exists(re_ad):
        os.makedirs(re_ad)
    print(re_ad)
    np.save(osp.join(re_ad, "{}_true_{}_{}_{}_{}.npy".format(style, sm_scale, name, m_train, m_test)), true)
    np.save(osp.join(re_ad, "{}_pred_{}_{}_{}_{}.npy".format(style, sm_scale, name, m_train, m_test)), pred)
    np.save(osp.join(re_ad, "{}_loss_{}_{}_{}_{}.npy".format(style, sm_scale, name, m_train, m_test)), np.array(loss))
    if model is not None:
        torch.save(model.state_dict(),
                   osp.join(re_ad, "model_{}_{}_{}_{}.pkl".format(sm_scale, name, m_train, m_test)))
    return True


def read_snr(df, style):
    snr = df.snr_db.values
    num = snr.shape[0]
    snr_all = []
    for i in range(num):
        snr_one = snr[i][1:-1]
        snr_one_ = snr_one.split(' ')
        num_full = snr_one_.count('')
        idx = 0
        while idx < num_full:
            snr_one_.remove('')
            idx = idx + 1
        snr_one_ = np.array([float(snr_one_[0]), float(snr_one_[1]), float(snr_one_[2])])
        if style == "mean":
            snr_one_mean = np.mean(snr_one_)
            snr_all.append(snr_one_mean)
        else:
            raise TypeError("Unknown type of style")
    snr_all = np.array(snr_all)
    return snr_all


def get_dist(feature, bins, chunk_name, data_size, v_min=None, v_max=None):
    df = pd.read_csv(osp.join(ROOT, chunk_name, chunk_name + ".csv"))
    if feature == "snr_db":
        data = read_snr(df, style="mean")
    else:
        data = df.loc[:, feature].values.reshape(-1)[:data_size - 1]

    if feature == "source_depth_km":
        data = data[data != "None"].astype(float)

    if v_min is not None:
        data = data[data >= v_min]
        data = np.append(data, v_min)
    if v_max is not None:
        data = data[data <= v_max]
        data = np.append(data, v_max)
    label, _ = pd.cut(data, bins=bins, retbins=True)
    label_vc = pd.DataFrame(label).value_counts()
    interval, y = label_vc.index.tolist(), label_vc.values
    x, left, right = [], float('inf'), -float('inf')
    for i in range(bins):
        interval_one = interval[i][0]
        left_one, right_one = interval_one.left, interval_one.right
        x.append((left_one + right_one) / 2)
        if left_one < left:
            left = left_one
        if right_one > right:
            right = right_one
    x = np.array(x)
    sort_index = np.argsort(x)
    x, y = x[sort_index], y[sort_index]
    if v_min is not None:
        y[0] = y[0] - 1
    if v_max is not None:
        y[-1] = y[-1] - 1
    return x, y


def be_numpy(x):
    if torch.is_tensor(x):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Unknown type of x, must be 'tensor' or 'ndarray'!")


class Chunk(Dataset):
    def __init__(self, data_size, train, data_size_train, idx, root, chunk_name):
        super(Chunk, self).__init__()
        self.data_size, self.root, self.name = data_size, root, chunk_name
        self.save_ad = osp.join(root, str(data_size))
        self.df = pd.read_csv(osp.join(self.root, self.name + ".csv"))
        self.data, self.index = self.get_sample()
        self.df = self.df.iloc[self.index, :]
        self.data_size_train = data_size_train
        self.length = self.data.shape[2]
        self.train = train
        self.idx = idx
        self.get_train_or_test()

    def get_train_or_test(self):
        self.data = self.data[self.idx, :, :]
        self.index = self.index[self.idx]
        self.df = self.df.iloc[self.idx, :]
        return None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data, index = self.data[idx, :, :], self.index[idx]
        return data, index

    def get_sample(self):
        if not osp.exists(self.save_ad):
            os.makedirs(self.save_ad)
        data_ad = osp.join(self.save_ad, "data.pt")
        index_ad = osp.join(self.save_ad, "index.pt")
        if osp.exists(data_ad) & osp.exists(index_ad):
            data = torch.load(data_ad)
            index = be_numpy(torch.load(index_ad))
        else:
            metadata = h5py.File(osp.join(self.root, self.name + ".hdf5"), 'r')

            trace_name = self.df.loc[:, "trace_name"].values.reshape(-1)
            index = np.random.choice(trace_name.shape[0], self.data_size, replace=False).tolist()

            ev_list = self.df['trace_name'].to_list()
            data = np.zeros(shape=(self.data_size, 3, 6000))
            for c, i in enumerate(index):
                ev_one = ev_list[i]
                dataset_one = metadata.get('data/' + str(ev_one))
                data_one = np.array(dataset_one)
                data_one = np.expand_dims(data_one.T, axis=0)
                data[c, :, :] = data_one

            data = torch.from_numpy(data).float()
            index = torch.FloatTensor(index).int()

            torch.save(data, data_ad)
            torch.save(index, index_ad)
        return data, be_numpy(index)


def get_mai_data(df_train, df_test):
    ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
    ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
    prep_ps_at, ps_at_train, ps_at_test = prep_pt("sta", ps_at_train, ps_at_test)
    ps_at_train, ps_at_test = torch.from_numpy(ps_at_train).float(), torch.from_numpy(ps_at_test).float()

    t_name = ["p_travel_sec"]
    p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
    prep_p_t, p_t_train, p_t_test = prep_pt("sta", p_t_train, p_t_test)
    p_t_train, p_t_test = torch.from_numpy(p_t_train).float(), torch.from_numpy(p_t_test).float()
    return ps_at_train, ps_at_test, p_t_train, p_t_test
