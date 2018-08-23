import layers
import numpy as np
from copy import deepcopy


def Train_n(Net, criterion, train_data, train_label, n, record=None):
    for k in range(1, n + 1):
        x = train_data
        for layer in Net:
            x = layer.forward(x)
        loss = criterion.forward(x, train_label)
        _ = loss
        grad = criterion.backprop()
        for layer in Net[::-1]:
            grad = layer.backprop(grad)
            if layer.need_update:
                layer.update()
        if record is not None and (n % record) == 0:
            print("After {}/{}, Loss: {}".format(k, n, loss))


def Train(train_data_full,
          train_label_full,
          test_data,
          test_label,
          activation,
          lr=0.05,
          dp_n=200,
          dp_p=10):

    idx = int(train_data_full.shape[0] * 0.8)
    train_data = train_data_full[:idx]
    train_label = train_label_full[:idx]
    valid_data = train_data_full[idx:]
    valid_label = train_label_full[idx:]

    W1 = np.random.randn(5, 13) / np.sqrt(6)
    b1 = np.zeros(13)
    W2 = np.random.randn(13, 4) / np.sqrt(6)
    b2 = np.zeros(4)

    fc1 = layers.FC(W1, b1, lr)
    act_f1 = layers.get_act_func(activation)
    fc2 = layers.FC(W2, b2, lr)

    criterion = layers.Softmax()

    Net = [fc1, act_f1, fc2]
    best_Net = deepcopy(Net)
    best_i = 0

    n = dp_n
    p = dp_p
    i = 0
    j = 0
    v = 99999  # very large number
    while j < p:
        # 01. TRAINING [n] STEPS
        Train_n(Net, criterion, train_data, train_label, n)

        # 02. Save or Try
        i = i + n
        vv = Test(valid_data, valid_label, Net)
        # print("Loop: {}, Acc: {}".format(i, 1 - vv))
        if vv < v:
            j = 0
            best_Net = deepcopy(Net)
            best_i = i
            v = vv
            print("New Best Point: {}, {}".format(best_i, 1 - vv))
        else:
            j += 1
            print("Try...[{}:{}]".format(j, 1 - vv))

    Test(test_data, test_label, best_Net, "[Test Set]")


def Test(test_data, test_label, _layers=None, name=None):
    if _layers is None:
        raise ValueError("Layers must be provieded!")
    else:
        layers = deepcopy(_layers)
        res = test_data
        for layer in layers:
            res = layer.forward(res)
        pred = np.argmax(res, 1)
        acc = round(np.mean(pred == test_label), 3)
        if name is not None:
            print("acc {} : {}".format(name, acc))
        del layers
        return 1 - acc
