import numpy as np
import pandas as pd


def bootstrapping(data, label, theresold=1.5):
    train_data = pd.DataFrame()
    train_label = []
    test_data = pd.DataFrame()
    test_label = []
    data_length = len(data)
    iter_times = int(data_length * theresold)
    boot_set = set([np.random.randint(0, data_length)
                    for i in range(iter_times)])
    boot_set_diff = set(range(data_length)).difference(boot_set)
    l_bs = list(boot_set)
    l_bs_d = list(boot_set_diff)
    train_data = data.iloc[l_bs]
    test_data = data.iloc[l_bs_d]
    train_label = label[l_bs]
    test_label = label[l_bs_d]
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    import data_prefix
    train_data_ori, train_label_ori = data_prefix.get_data("titanic/train.csv")
    train_data, train_label, test_data, test_label = bootstrapping(
        train_data_ori, train_label_ori)
    for i in[train_data, train_label, test_data, test_label]:
        print(len(i) / len(train_data_ori))
