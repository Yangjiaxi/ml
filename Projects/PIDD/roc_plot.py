import matplotlib.pyplot as plt
import numpy as np


def get_roc_points(roc_data):
    croods = [(0, 0)]
    crood = [0, 0]
    dlenP = 1 / len(roc_data[roc_data[:, 1] == 1])
    dlenN = 1 / len(roc_data[roc_data[:, 1] == 0])
    AUC = 0.0
    for i in range(len(roc_data)):
        if roc_data[i, 1] == 1:
            crood = [crood[0], crood[1] + dlenP]
        else:
            crood = [crood[0] + dlenN, crood[1]]
            AUC += float(dlenN) * crood[1]
        croods.append(crood)
    return np.array(croods), AUC


def plot_ROC(score, label, title):
    test_roc_data = np.column_stack((score.reshape((-1, 1)), label.reshape(-1, 1)))
    test_roc_data = test_roc_data[(0 - test_roc_data[:, 0]).argsort()]
    croods, AUC = get_roc_points(test_roc_data)
    crood_x, crood_y = croods[:, 0], croods[:, 1]
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.plot(crood_x, crood_y, color='blue', linewidth=1.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='red', linewidth=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.text(0.4, 0, "AUC : %f" % AUC, fontsize=25)
    plt.grid(color='red', linewidth=0.5, linestyle='-')
    plt.show()
