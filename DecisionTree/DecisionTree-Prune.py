import numpy as np
from collections import Counter
import pandas as pd
from TreePlotter import createPlot
import stratified_sampling as s_s


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


def calc_Gini(data):
    n = data.shape[0]
    data_labels = set(data[:, -1])
    labels_info = np.array(list(Counter(data[:, -1]).values()))
    gini = 1 - ((labels_info / n) ** 2).sum()
    return gini


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


def CART_split(data):
    feat_size = data.shape[1] - 1
    base_gini = calc_Gini(data)
    best_gini_gain = 0.0
    best_feat = -1
    for i in range(feat_size):
        values_on_feat = set(data[:, i])
        new_gini = 0.0
        for value in values_on_feat:
            sub_data_set = split_by_value(data, i, value)
            prob = sub_data_set.shape[0] * 1.0 / data.shape[0]
            new_gini += prob * calc_Gini(sub_data_set)
        gini_gain = base_gini - new_gini
        if (gini_gain > best_gini_gain):
            best_gini_gain = gini_gain
            best_feat = i
    return best_feat


def split_by_value(data, axis, value):
    data_f = data[data[:, axis] == value]
    data_r = np.delete(data_f, axis, 1)
    return data_r


def majority_vote(category_list):
    return Counter(category_list).most_common(1)[0][0]


def split(data, method):
    switcher = {
        "ID3": ID3_split,
        "C45": C45_split,
        "CART": CART_split
    }
    if method in switcher:
        return switcher[method](data)
    else:
        raise ValueError("No such split method called [ %s ]" % method)


def tree_build(data, features_ori, method):
    features = features_ori[:]
    category_list = data[:, -1]
    if Counter(category_list)[category_list[0]] == category_list.shape[0]:
        return category_list[0]
    if data.shape[0] == 1:
        return majority_vote(category_list)
    best_feature = split(data, method)
    best_feature_name = features[best_feature]
    tree_grow = {best_feature_name: {}}
    del (features[best_feature])
    values_on_feat = set(data[:, best_feature])
    for value in values_on_feat:
        sub_features = features[:]
        tree_grow[best_feature_name][value] = tree_build(
            split_by_value(data, best_feature, value), sub_features, method)
    return tree_grow


def data_split(data, ratio=0.3):
    train, test = s_s.stratified(data, ratio=ratio)
    return train, test


def classify(trained_tree, feature_names, test_sample):
    feature_key = list(trained_tree.keys())[0]
    if feature_key in trained_tree.keys():
        feature_value = trained_tree[feature_key]

        classify_res = 'acc'
        if feature_key in feature_names:
            feature_index = feature_names.index(feature_key)
            for key in feature_value.keys():
                if test_sample[feature_index] == key:
                    if isinstance(feature_value[key], dict):
                        classify_res = classify(
                            feature_value[key], feature_names, test_sample)
                    else:
                        classify_res = feature_value[key]
    return classify_res


def ori_count(tree, validation, features):  # error count of origin tree
    error = 0.0
    for i in range(len(validation)):
        res_i = classify(tree, features, validation[i])
        if res_i != validation[i][-1]:
            error += 1
    return float(error)


def pruning_count(major, validation):  # error count of pruning tree
    error = 0.0
    for i in range(len(validation)):
        if major != validation[i][-1]:
            error += 1
    return float(error)


def post_prune(tree, train, validation, features_ori):
    features = features_ori[:]
    feature_key = list(tree.keys())[0]
    feature_value = tree[feature_key]
    category_list = train[:, -1]
    feature_index = features.index(feature_key)
    del(features[feature_index])
    for key in feature_value.keys():
        if isinstance(feature_value[key], dict):
            sub_feat = features[:]
            tree[feature_key][key] = post_prune(
                feature_value[key],
                split_by_value(train, feature_index, key),
                split_by_value(validation, feature_index, key),
                sub_feat
            )

    ori_err = ori_count(tree, validation, features)
    new_err = pruning_count(majority_vote(category_list), validation)
    # print(new_err, ori_err)
    if ori_err * 0.1 < new_err:
        # print("Save!")
        return tree

    # print("Cut!!")
    return majority_vote(category_list)


def decision_tree(path):
    data_ori = pd.read_csv(path)
    data_ori = pd.read_csv(path)
    train_ori, test = data_split(data_ori, ratio=0.5)
    train = train_ori
    validation = test

    features = list(test.columns)
    tested_count = test.shape[0]
    best_err = 1.0
    best_tree = None
    best_method = None
    for method in ['ID3', 'C45', 'CART']:
        trained_tree = tree_build(np.array(train), list(train.columns), method)
        error_counts = 0
        for i in range(tested_count):
            i_info = list(test.iloc[i])
            i_data = i_info[:-1]
            i_label = i_info[-1]
            if not classify(trained_tree, features, i_data) == i_label:
                error_counts += 1

        err_rate = 1.0 * error_counts / test.shape[0]
        print("==================================")
        print("--->Based on %s split method" % method)
        print("%d samples tested\n   error rate is %f\naccuracy rate is %f"
              % (tested_count, err_rate, 1.0 - err_rate))
        if err_rate < best_err:
            best_err = err_rate
            best_tree = trained_tree
            best_method = method
    print("##################################")
    print("the best method for tree-building is [%s]" % best_method,
          "\nthe best accuracy is %f" % (1 - best_err))

    post_tree = post_prune(best_tree, np.array(
        train), np.array(validation), features)
    error_counts = 0
    for i in range(tested_count):
        i_info = list(test.iloc[i])
        i_data = i_info[:-1]
        i_label = i_info[-1]
        if not classify(post_tree, features, i_data) == i_label:
            error_counts += 1

    err_rate = 1.0 * error_counts / test.shape[0]
    print("==================================")
    print("--->Then, make post-pruning ")
    print("%d samples tested\n   error rate is %f\naccuracy rate is %f"
          % (tested_count, err_rate, 1.0 - err_rate))
    print("\ndict-based tree model has been returned ")
    return post_tree


if __name__ == '__main__':
    car_tree = decision_tree("car.csv")

'''
==================================
--->Based on ID3 split method
863 samples tested
   error rate is 0.081112
accuracy rate is 0.918888
==================================
--->Based on C45 split method
863 samples tested
   error rate is 0.070684
accuracy rate is 0.929316
==================================
--->Based on CART split method
863 samples tested
   error rate is 0.076477
accuracy rate is 0.923523
##################################
the best method for tree-building is [C45]
the best accuracy is 0.929316
==================================
--->Then, make post-pruning
863 samples tested
   error rate is 0.062572
accuracy rate is 0.937428

dict-based tree model has been returned
'''
