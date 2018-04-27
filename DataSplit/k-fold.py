import pandas as pd
import numpy as np


class K_fold_split:
    def __init__(self, data_ori, k=None):
        self.data_ori = data_ori
        data_ori_length = data_ori.shape[0]
        if k is not None:
            self.k = k
        else:
            self.k = data_ori_length
        self.size = int(data_ori_length / self.k)
        idx_drop = np.random.choice(
            data_ori_length, data_ori_length - self.k * self.size, replace=False)
        self.data = data_ori.iloc[list(
            set(list(range(data_ori_length))).difference(set(idx_drop.tolist())))]
        self.data_length = self.data.shape[0]
        self.idx_all = np.random.choice(
            self.data_length, self.data_length, replace=False)
        self.count = 0

    def next(self):
        if self.count < self.k:
            k_part = self.idx_all[self.count *
                                  self.size: (self.count + 1) * self.size].tolist()
            o_part = list(set(list(range(self.data_length))
                              ).difference(set(k_part)))
            self.count += 1
            return self.data.iloc[k_part], self.data.iloc[o_part]
        else:
            raise ValueError(
                "Loop is finish, please use .reset() to reboot the class")

    def get_list(self):
        return self.idx_all

    def reset(self):
        self.count = 0

    @property
    def turns(self):
        return self.k


if __name__ == '__main__':
    data_ori = pd.read_csv("car.csv")[0:10]
    kk_LOO = K_fold_split(data_ori)
    print("%d turns are loaded!" % kk_LOO.turns)
    for i in range(kk_LOO.turns):
        k, o = kk_LOO.next()
        print("->%02d==============================================" % i)
        print(k)
        print("-------------------------")
        print(o)
        print("==================================================")
    kk_LOO.reset()
