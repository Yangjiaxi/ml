import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from collections import Counter


def read_data():
    train_img_path = "mnist/train-images-idx3-ubyte"
    train_lbl_path = "mnist/train-labels-idx1-ubyte"
    test_img_path = "mnist/t10k-images-idx3-ubyte"
    test_lbl_path = "mnist/t10k-labels-idx1-ubyte"
    with open(train_lbl_path, 'rb') as train_label:
        magic, num = struct.unpack(">II", train_label.read(8))
        train_label = np.fromfile(train_label, dtype=np.int8)

    with open(test_lbl_path, 'rb') as test_label:
        magic, num = struct.unpack(">II", test_label.read(8))
        test_label = np.fromfile(test_label, dtype=np.int8)

    with open(train_img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_img = np.fromfile(fimg, dtype=np.uint8).reshape(
            len(train_label), -1) / 255

    with open(test_img_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_img = np.fromfile(fimg, dtype=np.uint8).reshape(
            len(test_label), -1) / 255

    return train_img, train_label, test_img, test_label, rows, cols


def plot_number(image, rows, cols):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image.reshape(rows, cols), cmap='gray')
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()


def classify_10(data, label, img, k=10):
    d_1 = np.abs(data - img)
    d_2 = d_1 ** 2
    d_3 = d_2.sum(axis=1)
    k_N = Counter(label[d_3.argsort()][:k])
    return sorted(k_N, key=lambda x: k_N[x])[-1]


def kNN(train_img, train_label, test_img, test_label, k=10):
    error_count = 0
    acc_rate = 1.0
    for i in range(len(test_img)):
        pred = classify_10(train_img, train_label, test_img[i])
        if pred != test_label[i]:
            error_count += 1
        acc_rate = 1 - 1.0 * error_count / (i + 1)
        print("(%05d / %d) => (%d, %d) => %s => %f" %
              (i + 1, len(test_img), pred, test_label[i], "Correct" if pred == test_label[i] else "*Error*", acc_rate))


if __name__ == '__main__':
    train_img, train_label, test_img, test_label, rows, cols = read_data()
    kNN(train_img, train_label, test_img, test_label)
