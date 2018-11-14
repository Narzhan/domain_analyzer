import pandas
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset = pandas.read_csv("test_data.csv", index_col=13)
for cat in list(dataset):
    x1 = list(dataset[dataset['label'] == 1][cat])
    x2 = list(dataset[dataset['label'] == 0][cat])
    colors = ["red", 'blue']
    names = ["škodlivé", "čisté"]
    plt.hist([x1, x2], color=colors, edgecolor="black", label=names, stacked=True, bins=15)
    plt.title(cat)
    plt.legend()
    plt.xlabel("Hodnota")
    plt.ylabel("Počet výsktů")
    plt.savefig("hist_{}.png".format(cat))
    plt.gcf().clear()
    plt.show()

