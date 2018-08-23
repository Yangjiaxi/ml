from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as cm
import numpy as np
import seaborn as sb
sb.set()
sb.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 144
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.round(np.sort([np.random.randn() for _ in range(101)]), 2)
Y = np.round(np.sort([np.random.randn() for _ in range(101)]), 2)
Z = np.round(np.sort([np.random.randn() for _ in range(101)]), 2)

for (a, b, c) in zip(X, Y, Z):
    print("({}, {}, {})".format(a, b, c))

cset = ax.plot(X, Y, Z)
ax.clabel(cset, fontsize=9, inline=1)
plt.show()
