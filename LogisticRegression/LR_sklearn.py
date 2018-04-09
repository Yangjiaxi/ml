import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fix_nan_all(data):
    return data.fillna(1)


def flatten(data):
    data = data + data.min(axis=1).reshape((-1, 1))
    return data / data.max(axis=1).reshape((-1, 1))


def load_file(file, method):
    data = pd.read_csv(file, header=None)
    values = fix_nan_all(data).values
    flattened = flatten(values[:, :-1].T)
    return np.row_stack((flattened, np.ones(flattened.shape[1]))), values[:, -1].reshape((1, -1))


if __name__ == '__main__':
    train_data, train_label = load_file("salted_fish/salted_fish_train.csv", 'train')
    test_data, test_label = load_file("salted_fish/salted_fish_test.csv", 'test')
    train_data = train_data.T
    test_data = test_data.T
    train_label = train_label.reshape((-1,))
    test_label = test_label.reshape((-1,))
    sc = StandardScaler()
    sc.fit(train_data)
    train_data_std = sc.fit(train_data)
    test_data_std = sc.fit(test_data)
    lr = LogisticRegression(C=1.0, penalty='l2', max_iter=100000, tol=0.009)
    lr.fit(train_data, train_label)
    test_prediction = lr.predict(test_data)
    test_score = lr.score(test_data, test_label)
    print(classification_report(test_label, test_prediction))
    print(test_score)
