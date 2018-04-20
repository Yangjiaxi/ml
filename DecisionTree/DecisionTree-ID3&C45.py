import numpy as np
from collections import Counter
import pandas as pd


def calc_ent(data):
    labels = Counter(data[:, -1])
    values = np.array(list(labels.values()))
    values_prob = values / values.sum()
    ent = -(values_prob * np.log2(values_prob)).sum()
    return ent


def calc_IV(data, axis):
    labels = Counter(data[:, axis])
    values = np.array(list(labels.values()))
    values_prob = values / values.sum()
    ent = -(values_prob * np.log2(values_prob)).sum()
    return ent


def split_by_value(data, axis, value):
    data_f = data[data[:, axis] == value]
    data_r = np.delete(data_f, axis, 1)
    return data_r


def ID3_split(data):
    feat_size = data.shape[1] - 1
    base_ent = calc_ent(data)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(feat_size):
        values_on_feat = set(data[:, i])
        new_ent = 0.0
        for value in values_on_feat:
            sub_data_set = split_by_value(data, i, value)
            prob = sub_data_set.shape[0] * 1.0 / data.shape[0]
            new_ent += prob * calc_ent(sub_data_set)
        info_gain = base_ent - new_ent
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feat = i
    return best_feat


def C45_split(data):
    feat_size = data.shape[1] - 1
    base_ent = calc_ent(data)
    best_info_gain_rate = 0.0
    best_feat = -1
    for i in range(feat_size):
        values_on_feat = set(data[:, i])
        new_ent = 0.0
        for value in values_on_feat:
            sub_data_set = split_by_value(data, i, value)
            prob = sub_data_set.shape[0] * 1.0 / data.shape[0]
            new_ent += prob * calc_ent(sub_data_set)
        info_gain_rate = (base_ent - new_ent) / calc_IV(data, i)
        if (info_gain_rate > best_info_gain_rate):
            best_info_gain_rate = info_gain_rate
            best_feat = i
    return best_feat


def majority_vote(category_list):
    return Counter(category_list).most_common(1)[0][0]


def tree_build(data, features_ori, method):
    features = features_ori[:]
    category_list = data[:, -1]
    if Counter(category_list)[category_list[0]] == category_list.shape[0]:
        return category_list[0]
    if data.shape[0] == 1:
        return majority_vote(category_list)
    if method == 'ID3':
        best_feature = ID3_split(data)
    elif method == 'C45':
        best_feature = C45_split(data)
    else:
        raise ValueError("No such split method called [ %s ]" % method)
    # print("best: ", bestFeat)
    best_feature_name = features[best_feature]
    tree_grow = {best_feature_name: {}}
    del (features[best_feature])
    values_on_feat = set(data[:, best_feature])
    for value in values_on_feat:
        sub_features = features[:]
        tree_grow[best_feature_name][value] = tree_build(
            split_by_value(data, best_feature, value), sub_features, method)
    return tree_grow


def classify(trained_tree, feature_names, test_sample):
    feature_key = list(trained_tree.keys())[0]
    feature_value = trained_tree[feature_key]
    feature_index = feature_names.index(feature_key)
    classify_res = 'Unknown'
    for key in feature_value.keys():
        if test_sample[feature_index] == key:
            if isinstance(feature_value[key], dict):
                classify_res = classify(
                    feature_value[key], feature_names, test_sample)
            else:
                classify_res = feature_value[key]
    return classify_res


def data_split(data):
    import bootstrapping as bs
    _label = np.zeros(data.shape[0])
    train, _, test, _ = bs.bootstrapping(data, _label)
    return train, test


def decision_tree(path, method="ID3"):
    data_ori = pd.read_csv(path)
    train, test = data_split(data_ori)
    car_tree = tree_build(np.array(train), list(train.columns), method)
    features = list(test.columns)
    error_counts = 0
    tested_count = test.shape[0]
    for i in range(tested_count):
        i_info = list(test.iloc[i])
        i_data = i_info[:-1]
        i_label = i_info[-1]
        if not classify(car_tree, features, i_data) == i_label:
            error_counts += 1
        # print("error!, at ", i)

    err_rate = 1.0 * error_counts / test.shape[0]
    print("##Based on %s split method##" % method)
    print("----------------------------------------")
    print("%d samples tested\nerror rate is %f\n  acc rate is %f"
          % (tested_count, err_rate, 1.0 - err_rate))
    return car_tree


if __name__ == '__main__':
    car_tree = decision_tree("car.csv", method='C45')
