import os
from utils import load_file
from bp import Train

import numpy as np

ss = os.getcwd()
path = os.path.join(ss, "student.xls")
train_data, train_label, test_data, test_label, lvl_map = load_file(path)
Train(train_data,
      train_label,
      test_data,
      test_label,
      epochs=5000,
      batch_size=64,
      lr=0.1,
      decay=0.99,
      epochs_drop=1000,
      activation="l_relu")
