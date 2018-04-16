import numpy as np
import matplotlib.pyplot as plt
import os


def read_data(path):
    data_str = []
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            data_str.append(line.split())
    return np.array(data_str).astype(np.float64)


def plot_res(C_set, k):
    color_dict = ['red',
                  'blue',
                  'yellow',
                  'purple',
                  'pink']
    for i in range(k):
        plt.scatter(np.array(C_set[i])[:, 0],
                    np.array(C_set[i])[:, 1],
                    color=color_dict[i])
    plt.scatter(
        np.array(C_set[-1])[:, 0],
        np.array(C_set[-1])[:, -1],
        color='black',
        marker='x')
    plt.show()


def run(data_path, k):
    data_set = read_data(data_path)
    m = data_set.shape[0]
    mean_vector = data_set[np.random.choice(m, k, replace=False)]
    Flag = True
    while Flag:
        C = []
        for i in range(k + 1):
            C.append([])
        for j in range(m):
            dj = ((data_set[j] - mean_vector) ** 2).sum(axis=1).argmin()
            C[dj].append(data_set[j].tolist())
        mean_prev = mean_vector.copy()
        for i in range(k):
            mean_vector[i] = np.array(C[i]).mean(axis=0).tolist()
        mean_delta = np.min(np.abs(mean_prev - mean_vector))
        if mean_delta == 0.0:
            Flag = False
    C[k] = mean_vector[:].tolist()
    plot_res(C, k)


if __name__ == '__main__':
    run('watermelon_4.0.txt', 3)
