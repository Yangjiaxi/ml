import pandas as pd
import numpy as np


def stratified(data, ratio=0.3):
    sorts = list(set(data_ori.iloc[:, -1]))
    test = pd.DataFrame()
    for sort in sorts:
        idx = (data.iloc[:, -1] == sort)
        i_sort = data[idx]
        i_len = idx.sum()
        i_target = i_len * ratio
        get = i_sort.iloc[np.random.choice(
            i_len, int(i_len * ratio), replace=False)]
        # print(i_len, get.shape[0])
        test = pd.concat([test, get])

    train = data.iloc[list(set(data.index).difference(set(test.index)))]
    return train, test


if __name__ == "__main__":
    data_ori = pd.read_csv("car.csv")
    train, test = stratified(data_ori, ratio=0.3)
    print(train.shape[0], test.shape[0])
