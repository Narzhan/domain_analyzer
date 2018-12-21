import pandas
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.mlab as mlab
import csv, ast
from statistics import mean, stdev

dataset = pandas.read_csv("test_data.csv")


def heat_map():
    # dataset = dataset.drop('label', 1)
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
        if cat in ["totalEstimatedMatches"]:

            x1 = list(dataset[dataset['label'] == 1][cat])
            x2 = list(dataset[dataset['label'] == 0][cat])
            colors = ["red", 'blue']
            names = ["škodlivé", "čisté"]
            fig = plt.figure(figsize=(5.5, 4.0))
            plt.hist([x1, x2], color=colors, edgecolor="black", label=names, stacked=True
                    )
            plt.title(cat)
            plt.legend()
            plt.xlabel("Hodnota")
            plt.ylabel("Počet výsktů")
            # plt.savefig("hist_{}.png".format(cat))
            # plt.gcf().clear()
            plt.show()


def total_est_mtach_hist():
    cat = "totalEstimatedMatches"
    x1 = list(dataset[dataset['label'] == 1][cat])
    x2 = list(dataset[dataset['label'] == 0][cat])
    colors = ["red", 'blue']
    names = ["škodlivé", "čisté"]
    x2_formatted, x1_formatted = [], []
    for x in x2:
        if x < 8300:
            x2_formatted.append(x)
        else:
            x2_formatted.append(8300)
    for x in x1:
        if x < 8300:
            x1_formatted.append(x)
        else:
            x1_formatted.append(8300)
    fig = plt.figure(figsize=(5.5, 4.0))
    plt.hist([x1_formatted, x2_formatted], color=colors, edgecolor="black", label=names, stacked=True
             )
    plt.title(cat)
    plt.legend()
    plt.xlabel("Hodnota")
    plt.ylabel("Počet výsktů")
    plt.savefig("hist_{}.png".format(cat))
    plt.gcf().clear()
    # plt.show()


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

def compare_graph():
    results=[]
    names=[]
    for k in range(5,11):
        with open("variable substition/result_first_test{}.txt".format(k), "r") as file:
            reader =csv.reader(file)
            next(reader)
            next(reader)
            for row in reader:
                names.append(row[0])
                row[1]=",".join(row[1].split(" "))
                results.append(ast.literal_eval(row[1]))
        print(k)
        for name, score in zip(names, results):
            print("{}: {} ({})".format(name, mean(score), stdev(score)))
        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111)
        plt.boxplot(results, 0, '')
        ax.set_xticklabels(names)
        plt.ylabel("Přesnost")
        for tick in ax.get_xticklabels():
            tick.set_rotation(70)
        plt.savefig("test_data{}png".format(k))
        results.clear()
        names.clear()


def outliers_box_plot():
    l = dataset.columns.values
    number_of_columns = 12
    number_of_rows = len(l) - 1 / number_of_columns
    plt.figure(figsize=(number_of_columns, 5 * number_of_rows))
    for i in range(0, len(l)):
        plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
        sns.set_style('whitegrid')
        sns.boxplot(dataset[l[i]], color='green', orient='v')
        plt.tight_layout()
    plt.show()


def skewness():
    l = dataset.columns.values
    number_of_columns = 12
    number_of_rows = len(l) - 1 / number_of_columns
    plt.figure(figsize=(2 * number_of_columns, 5 * number_of_rows))
    for i in range(0, len(l)):
        plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
        sns.distplot(dataset[l[i]], kde=True)
    plt.show()


# # outliers_box_plot()
# skewness()
import csv

# with open("text_test_data.csv", "r", errors="ignore") as file:
#     with open("text_test_data_english.csv", "w", errors="ignore") as output:
#         reader = csv.reader(file, delimiter=";")
#         output.write("{}\n".format(";".join(next(reader))))
#         for row in reader:
#             output.write("{}\n".format(";".join(row)))