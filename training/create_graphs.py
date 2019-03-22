import pandas
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.mlab as mlab
import csv, ast, operator
from statistics import mean, stdev

dataset = pandas.read_csv("dataframe_enhanced.csv", index_col=0)
# sns.pairplot(dataset, hue='label')

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
        if cat not in ["topics", "tf_idf", "embedding",
               'label', "domain"]:
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
            plt.savefig("hists/hist_{}.png".format(cat))
            plt.gcf().clear()
            # plt.show()


def total_est_mtach_hist():
    cat = "totalEstimatedMatches"
    print(cat)
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
    plt.savefig("hists/hist_{}.png".format(cat))
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


def langugae_distribution():
    lang_count = {'ms': 1275, 'sv': 6116, 'sk': 3869, 'ka': 353, 'mk': 402, 'lt': 1205, 'nb': 1, 'la': 257, 'nn': 96,
                  'is': 415, 'so': 77, 'cs': 59516, 'bs': 33, 'pt': 50177, 'bn': 1241, 'si': 135, 'tr': 17630,
                  'en': 1503042, 'gu': 99, 'sl': 1406, 'nl': 14570, 'bi': 1, 'sw': 94, 'co': 2, 'hy': 263, 'ro': 6797,
                  'or': 6, 'ja': 68944, 'he': 1537, 'mn': 1, 'te': 287, 'vi': 9739, 'cy': 156, 'zh_chs': 57404,
                  'sr': 2464, 'ts': 1, 'jv': 9, 'sq': 578, 'ia': 6, 'id': 12368, 'el': 7193, 'pa': 24, 'dv': 4,
                  'fi': 4264, 'mg': 2, 'lo': 25, 'km': 185, 'mt': 13, 'af': 98, 'ga': 22, 'kk': 3, 'az': 61, 'lb': 2,
                  'uk': 3157, 'ml': 290, 'pl': 21060, 'ur': 256, 'tl': 247, 'su': 2, 'eo': 34, 'ta': 693, 'da': 4223,
                  'ht': 4, 'fa': 21154, 'my': 318, 'aa': 22, 'ru': 88082, 'ca': 1633, 'uz': 362, 'ps': 44, 'be': 57,
                  'no': 3733, 'fy': 1, 'es': 80195, 'zh_cht': 14120, 'hu': 4727, 'hr': 2037, 'it': 39938, 'ii': 1,
                  'fr': 76044, 'th': 6991, 'an': 1, 'ar': 12523, 'sn': 1, 'eu': 199, 'kn': 111, 'ko': 17812, 'et': 587,
                  'gl': 215, 'bg': 2533, 'de': 69192, 'ky': 2, 'fo': 3, 'ku': 145, 'hi': 2470, 'lv': 555}
    print(sum(lang_count.values()), len(lang_count))
    lang_count = sorted(lang_count.items(), key=operator.itemgetter(1))
    lang_count = lang_count[len(lang_count)-20:]
    print(lang_count)
    # langs= []
    # counts=[]
    # for langugae in lang_count:
    #     langs.append(langugae[0])
    #     counts.append(langugae[1])
    # sns.set(style="whitegrid")
    # ax=sns.barplot(x=langs, y=counts)
    # plt.show()
from statistics import median

def plot_coherence():
    coherence_values = [-4.797301092927819, -3.963008547611659, -4.972567367283015, -5.351131376341533,
                        -4.338216212541027,
                        -5.210745925560019, -5.2297786159645225, -4.392520352889975, -5.458278450687757,
                        -4.697975245506327,
                        -5.480077815398153,
                        -4.822231370498071, -4.97530794907234, -5.483165586827279, -5.43950102560758,
                        -4.967268508371155,
                        -5.097836345708665, -5.902612497151679, -4.812677343844594, -5.715736750889206,
                        -5.272727025095471,
                        -5.636415965103936, -5.3676849859041, -5.977049925519246, -5.809777787445271,
                        -6.527556205148542, -6.469163867568904, -6.571937542866436, -6.607958664797175,
                        -6.771573127877327, -7.183241000622612, -7.517588205113764, -9.453467792944165]
    # coherence_values = coherence_values[:-8]
    number_of_topics = [40, 50, 60, 70, 80, 90, 100, 150]
    previous = [i for i in range(5,30)]
    previous.extend(number_of_topics)
    fig, ax = plt.subplots()
    ax.plot(previous, coherence_values, marker='o', color='b')
    # ax.xaxis.set_ticks(previous)
    # sum(coherence_values) / len(coherence_values)
    # plt.axhline(y=median(coherence_values))
    ax.set(xlabel='number of topics', ylabel='coherence',
           title='The change of coherence with the number of topics')
    ax.grid()
    plt.gca().invert_yaxis()
    # fig.savefig("test.png")
    plt.show()


# total_est_mtach_hist()