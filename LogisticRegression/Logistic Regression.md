# 逻辑回归

## 1. 模型参数

```python
d = model(train_data,        # 训练集数据
          train_label,       # 训练集标签
          test_data,         # 测试集数据
          test_label,        # 测试集标签
          [iter_times=10000, # 迭代次数[opt]
          alpha=0.01,        # 学习率[opt]
          C=1.0,             # 正则化系数[opt]
          show_epoch=False]) # 是否显示损失函数记录[opt]
```

## 2. 返回值：`dict对象`

```python
d[“costs”]             # list，每1000轮后的cost函数值          
d[“test_prediction”]   # list，测试集的预测结果
d[“train_prediction”]  # list，训练集的预测结果
d[“test_score”]        # list，测试集的得分(probability)
d[“train_score”]       # list，训练集的得分(probability)
d[“train_acc”]         # float，训练集精度
d[“test_acc”]          # float，测试集精度
d[“theta”]             # numpy.ndarray，习得参数d[“learning_rate”]     # int，学习率
d[“num_iterations”]    # int，迭代次数
```

## 3. 绘制ROC曲线

`from roc_plot import plot_ROC`

1. 参数：

   `plot_ROC(score, label, title)`

   - `score` 预测得分，无需排序
   - `label`对应标签集
   - `title` 图表标题

2. 返回：直接显示`matplotlib.pyplot.plot`曲线



