import layers
import numpy as np
from copy import deepcopy


def Train(train_data,
          train_label,
          test_data,
          test_label,
          epochs,
          activation,
          lr=0.05,
          decay=1,
          epochs_drop=1000,
          result_require=False):

    W1 = np.random.randn(5, 16) / np.sqrt(6)
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 31) / np.sqrt(6)
    b2 = np.zeros(31)
    W3 = np.random.randn(31, 11) / np.sqrt(6)
    b3 = np.random.randn(11)
    W4 = np.random.randn(11, 4) / np.sqrt(6)
    b4 = np.zeros(4)

    fc1 = layers.FC(W1, b1, lr, decay, epochs_drop)
    act_f1 = layers.get_act_func(activation)
    fc2 = layers.FC(W2, b2, lr, decay, epochs_drop)
    act_f2 = layers.get_act_func(activation)
    fc3 = layers.FC(W3, b3, lr, decay, epochs_drop)
    act_f3 = layers.get_act_func(activation)
    fc4 = layers.FC(W4, b4, lr, decay, epochs_drop)

    criterion = layers.Softmax()

    Net = [fc1, act_f1, fc2, act_f2, fc3, act_f3, fc4]

    # TRAINING BEGIN
    for i in range(1, epochs + 1):
        x = train_data
        for layer in Net:
            x = layer.forward(x)
        loss = criterion.forward(x, train_label)

        if i % (epochs / 5) == 0:
            print("After %d/%d epochs, loss : %f" % (i, epochs, loss))
            Test(train_data, train_label, Net, "[Train Set]")
            Test(test_data, test_label, Net, "[Test Set]")

        grad = criterion.backprop()
        for layer in Net[::-1]:
            grad = layer.backprop(grad)
            if layer.need_update:
                layer.update()
    # TRAINING FINISH

    pred = Test(test_data, test_label, Net, "[Test Set]")

    if result_require == True:
        return pred


def Test(test_data, test_label, _layers=None, name=None):
    if _layers is None:
        raise ValueError("Layers must be provieded!")
    else:
        layers = deepcopy(_layers)
        res = test_data
        for layer in layers:
            res = layer.forward(res)
        pred = np.argmax(res, 1)
        acc = np.mean(pred == test_label)
        print("acc {} : {}".format(name, acc))
        del layers
        return pred
