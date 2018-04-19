import numpy as np
import pandas as pd


def read_data(path):
    data_ori = pd.read_csv("watermelon_3.0.csv")
    col_map = {}
    for col in data_ori.columns[:-1]:
        col_set = set(data_ori[col])
        if len(col_set) <= data_ori.shape[0] * 0.85:
            col_dict = {}
            i = 0
            for ele in col_set:
                col_dict[ele] = i
                i += 1
            col_map[col] = col_dict
    col_map[data_ori.columns[-1]] = {"是": 1, "否": 0}
    data_fix = data_ori[:]
    for col in data_ori.columns:
        if col in col_map:
            data_fix[col] = data_fix[col].map(col_map[col])
    return data_fix.drop(['编号'], axis=1), col_map


def make_prob(data, data_map):
    data_length = data.shape[0]
    pos_prior = ((data['好瓜'] == data_map['好瓜']['是']).sum() +
                 1) / (data_length+len(data_map['好瓜']))
    neg_prior = ((data['好瓜'] == data_map['好瓜']['否']).sum() +
                 1) / (data_length+len(data_map['好瓜']))
    pos_num = (data['好瓜'] == data_map['好瓜']['是']).sum()
    neg_num = (data['好瓜'] == data_map['好瓜']['否']).sum()
    pos_map = {}
    neg_map = {}
    for cols in data.columns:
        if cols == '好瓜':
            pass
        else:
            pos_map[cols] = {}
            neg_map[cols] = {}
            if cols in data_map:
                for feats in data_map[cols]:
                    pos_f = ((data[data[cols] == data_map[cols][feats]]['好瓜'] == data_map['好瓜']['是']).sum(
                    )+1) / (pos_num + len(data_map[cols]))
                    neg_f = ((data[data[cols] == data_map[cols][feats]]['好瓜'] == data_map['好瓜']['否']).sum(
                    )+1) / (neg_num + len(data_map[cols]))
                    pos_map[cols][feats] = round(pos_f, 3)
                    neg_map[cols][feats] = round(neg_f, 3)
            else:
                pos_map[cols]['mean'] = round(
                    data[data['好瓜'] == 1][cols].mean(), 3)
                neg_map[cols]['mean'] = round(
                    data[data['好瓜'] == 0][cols].mean(), 3)
                pos_map[cols]['std'] = round(
                    data[data['好瓜'] == 1][cols].std(), 3)
                neg_map[cols]['std'] = round(
                    data[data['好瓜'] == 0][cols].std(), 3)
    return pos_prior, neg_prior, pos_map, neg_map


def pred(pos_prior, neg_prior, pos_map, neg_map, test):
    pos_prob = pos_prior
    neg_prob = neg_prior
    for cols in test:
        if 'mean' in pos_map[cols]:  # 是连续值属性
            alpha = (np.sqrt(2*np.pi)) * pos_map[cols]['std']
            beta = (test[cols][0] - pos_map[cols]['mean']
                    ) ** 2 / (2*pos_map[cols]['std'] ** 2)
            pos_feat_prob = 1 / alpha * np.exp(-beta)
            pos_prob *= pos_feat_prob

            alpha = (np.sqrt(2*np.pi)) * neg_map[cols]['std']
            beta = (test[cols][0] - neg_map[cols]['mean']
                    ) ** 2 / (2*neg_map[cols]['std'] ** 2)
            neg_feat_prob = 1 / alpha * np.exp(-beta)
            neg_prob *= neg_feat_prob
        else:
            pos_prob *= pos_map[cols][test[cols][0]]
            neg_prob *= neg_map[cols][test[cols][0]]
    pos_prob = round(pos_prob, 8)
    neg_prob = round(neg_prob, 8)
    return pos_prob, neg_prob, '是' if pos_prob > neg_prob else '否'


if __name__ == '__main__':
    data, data_map = read_data("watermelon_3.0.csv")
    pos_prior, neg_prior, pos_map, neg_map = make_prob(data, data_map)
    test_sample = pd.DataFrame(
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]).T
    test_sample.columns = data.columns[: - 1]
    print(pred(pos_prior, neg_prior, pos_map, neg_map, test_sample))
