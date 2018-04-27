import numpy as np
import pandas as pd


def bootstrapping(data, ratio=0.3):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    data_length = len(data)
    threshold = np.log(ratio) / (data_length * np.log(1 - 1 / data_length))
    iter_times = int(data_length * threshold)
    boot_set = set([np.random.randint(0, data_length)
                    for i in range(iter_times)])
    boot_set_diff = set(range(data_length)).difference(boot_set)
    l_bs = list(boot_set)
    l_bs_d = list(boot_set_diff)
    train_data = data.iloc[l_bs]
    test_data = data.iloc[l_bs_d]
    return train_data, test_data


if __name__ == '__main__':
    data_ori = pd.read_csv("car.csv")
    train, test = bootstrapping(data_ori, ratio=0.3)
    for item in [train, test]:
        print(item.shape[0] / data_ori.shape[0])
