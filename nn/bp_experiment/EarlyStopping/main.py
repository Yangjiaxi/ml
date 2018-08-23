import os
from utils import load_file
from bp import Train

ss = os.getcwd()
path = os.path.join(ss, "student.xls")
train_data, train_label, test_data, test_label, lvl_map = load_file(path)
Train(train_data,
      train_label,
      test_data,
      test_label,
      lr=0.021,
      activation="relu",
      dp_n=600,
      dp_p=5)

'''
New Best Point: 600, 0.596
New Best Point: 1200, 0.731
New Best Point: 1800, 0.827
New Best Point: 2400, 0.904
Try...[1:0.904]
Try...[2:0.904]
Try...[3:0.904]
New Best Point: 4800, 0.923
Try...[1:0.923]
Try...[2:0.923]
Try...[3:0.923]
Try...[4:0.923]
New Best Point: 7800, 0.942
Try...[1:0.942]
Try...[2:0.942]
Try...[3:0.942]
Try...[4:0.942]
Try...[5:0.942]
acc [Test Set] : 0.993
'''
