import numpy as np
import pandas as pd
from collections import Counter
import os


def get_data(path, theresold=0.95):
    data_ori = pd.read_csv(path)
    female_age_mean = data_ori[data_ori.Sex == 'female'].Age.mean()
    male_age_mean = data_ori[data_ori.Sex == 'male'].Age.mean()
    fixed_female = data_ori.loc[data_ori['Sex']
                                == 'female'].Age.fillna(female_age_mean)
    fixed_male = data_ori.loc[data_ori['Sex']
                              == 'male'].Age.fillna(male_age_mean)
    data_ori.loc[data_ori['Sex'] == 'female', 'Age'] = fixed_female
    data_ori.loc[data_ori['Sex'] == 'male', 'Age'] = fixed_male
    label = data_ori.Survived
    data_ori = data_ori.drop(['Survived', 'Cabin'], axis=1)
    data_ori.Sex = np.where(data_ori.Sex == 'male', 1, 0)
    data = pd.DataFrame()
    data_length = len(data_ori)
    for cols in data_ori:
        counter_dict = Counter(data_ori[cols])
        if len(counter_dict) < theresold * data_length:
            data[cols] = data_ori[cols]
            if str(data_ori[cols].dtypes) == 'object':
                it = 0
                counter_dict_index = dict()
                for k in Counter(data_ori[cols]).items():
                    counter_dict_index[k[0]] = it
                    it += 1
                data[cols].replace(counter_dict_index, inplace=True)
                del counter_dict_index
            data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()
    return data, np.array(label)


if __name__ == "__main__":
    data, label = get_data("titanic/train.csv")
    print(data.head())
    print(label[:5])
