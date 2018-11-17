import pandas
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset = pandas.read_csv("test_data.csv")


def heat_map():
    dataset = dataset.drop('label', 1)
    corr = dataset.corr()
    fig = plt.figure(figsize=(15.0, 25.0))
    g=sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, vmax=1, square=True,annot=True,cmap='cubehelix')
    #plt.yticks(rotation=30) 
    plt.xticks(rotation=40) 
    plt.show()

def histos():
    for cat in list(dataset):
        x1 = list(dataset[dataset['label'] == 1][cat])
        x2 = list(dataset[dataset['label'] == 0][cat])
        colors = ["red", 'blue']
        names = ["škodlivé", "čisté"]
        fig = plt.figure(figsize=(5.5, 4.0))
        plt.bar([[x1], [x2]], color=colors, edgecolor="black", label=names, stacked=True, bins=10)
        plt.title(cat)
        plt.legend()
        plt.xlabel("Hodnota")
        plt.ylabel("Počet výsktů")
        plt.savefig("hist_{}.png".format(cat))
        plt.gcf().clear()
        #plt.show()

def pairplot():
    fig = plt.figure(figsize=(25.0, 25.0))
    #sns.pairplot(dataset, hue="label")
    scatter_matrix(dataset, c=dataset["label"])
    plt.savefig("test.png")

def scatter():
    #iris["ID"] = iris.index
    #iris["ratio"] = iris["sepal_length"]/iris["sepal_width"]
    sns.lmplot(dataset.index.values, "page_count", data=dataset, hue="label", fit_reg=False, legend=True)
    plt.legend()
    plt.show()
scatter()
