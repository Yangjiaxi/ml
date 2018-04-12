import numpy as np
import pandas as pd
import os


def get_data(path, theresold=0.95):
    print(path)
    data_ori = pd.read_csv(path)
    female_age_mean = data_ori[data_ori.Sex == 'female'].Age.mean()
    male_age_mean = data_ori[data_ori.Sex == 'male'].Age.mean()
    fixed_female = data_ori.loc[data_ori['Sex']
                                == 'female'].Age.fillna(female_age_mean)
    fixed_male = data_ori.loc[data_ori['Sex']
                              == 'male'].Age.fillna(male_age_mean)
    data_ori.loc[data_ori['Sex'] == 'female', 'Age'] = fixed_female
    data_ori.loc[data_ori['Sex'] == 'male', 'Age'] = fixed_male
    data_ori.drop(['Survived', 'Cabin'], axis=1)
    return data_ori


if __name__ == '__main__':
    path = ""
    print("****************")
    # print(path)
    print(get_data(path))
