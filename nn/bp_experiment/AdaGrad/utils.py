import pandas as pd
import numpy as np


def load_file(file):
    lvl_map = {
        "High": 0,
        "Middle": 1,
        "Low": 2,
        "very_low": 3,
        "Very Low": 3
    }
    train_ori = pd.read_excel(file, sheet_name="Training_Data")
    test_ori = pd.read_excel(file, sheet_name="Test_Data")
    train_label = np.array(
        train_ori.iloc[:, -1].map(lvl_map)).astype(np.int).reshape((-1, ))
    test_label = np.array(
        test_ori.iloc[:, -1].map(lvl_map)).astype(np.int).reshape((-1, ))
    train_data = np.delete(np.array(train_ori), -1, -1).astype(np.float)
    test_data = np.delete(np.array(test_ori), -1, -1).astype(np.float)
    return train_data, train_label, test_data, test_label, lvl_map


def make_parm(m, n):
    low = -np.sqrt(6 / (m + n))
    high = -low
    return np.random.uniform(low, high, (m, n))
