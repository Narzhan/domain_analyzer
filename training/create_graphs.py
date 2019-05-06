import ast
import csv
import operator
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset = pandas.read_csv("dataframe_enhanced.csv", index_col=0)
del dataset["tf_idf"]
del dataset["topics"]
del dataset["embedding"]
# sns.pairplot(dataset, hue='label')


def reduced_data_pca():
    array = dataset.values
    X = array[:, 0:-1]
    Y = array[:, -1]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    from mpl_toolkits.mplot3d import Axes3D
    Axes3D = Axes3D
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)
    # raise Exception
    principalDf = pandas.DataFrame(data=principalComponents, index=dataset.index
                               , columns=['principal component 1', 'principal component 2', 'principal component 3'])
    principalDf = principalDf.join(dataset["label"])
    print(principalDf)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = principalDf[principalDf["label"]==1]['principal component 1']
    ys = principalDf[principalDf["label"]==1]['principal component 2']
    zs = principalDf[principalDf["label"]==1]['principal component 3']

    xt = principalDf[principalDf["label"]==0]['principal component 1']
    yt = principalDf[principalDf["label"]==0]['principal component 2']
    zt = principalDf[principalDf["label"]==0]['principal component 3']
    # ax.scatter(xs, ys, zs, s=50, alpha=0.6, label=dataset["label"], edgecolors='w')
    ax.scatter(xs, ys, zs, c='orange', s=50, alpha=0.6, edgecolors='w')
    ax.scatter(xt, yt, zt, c='b', s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel('principal component 1')
    ax.set_ylabel('principal component 2')
    ax.set_zlabel('principal component 3')

    plt.show()
    # ax = sns.scatterplot(x="principal component 1", y="principal component 2", data=principalDf, hue="label")
    # plt.show()


def reduced_data_lda():
    array = dataset.values
    X = array[:, 0:-1]
    Y = array[:, -1]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, Y)
    # print(X_lda)
    print(lda.explained_variance_ratio_)
    principalDf = pandas.DataFrame(data=X_lda, index=dataset.index
                               , columns=['principal component 1'])
    principalDf = principalDf.join(dataset["label"])
    x = sns.heatmap(principalDf, vmin=0, vmax=1)
    # sns.violinplot(x="label", y="principal component 1", data=principalDf)
    plt.show()

def pair_plot():
    g = sns.pairplot(dataset, hue="label")
    plt.show()

def heat_map():
    # dataset = dataset.drop('label', 1) ,cmap='cubehelix'
    corr = dataset.corr()
    fig = plt.figure(figsize=(15.0, 25.0))
    g=sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, vmax=1, square=True,annot=True)
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


def probability_graphs():
    x_label = [0.5, 0.6, 0.7, 0.8, 0.9]
    knn = [0.832, 0.832, 0.842, 0.842, 0.89]
    frorest = [0.865, 0.879, 0.898, 0.908, 0.914]
    lightgbm = [0.868, 0.888, 0.902, 0.916, 0.919]
    nn = [0.91, 0.911, 0.916, 0.922, 0.923]
    fig, ax = plt.subplots()
    ax.plot(x_label, knn, marker='o', color='b', label="knn")
    ax.plot(x_label, frorest, marker='o', color='r', label="rforests")
    ax.plot(x_label, lightgbm, marker='o', color='g', label="lightgbm")
    ax.plot(x_label, nn, marker='o', color='y', label="nn")
    ax.set(xlabel='probability threshold', ylabel='accuracy',
           title='Effect of probability threshold for prediction')
    ax.grid()
    plt.legend(loc='best')
    # fig.savefig("test.png")
    plt.show()


def non_zero_values():
    data = {"related_searches": 151,
            "full_path": 79608,
            "part_path": 193005,
            "about": 75518,
            "deep_links": 106310,
            "fresh": 95495,
            "infection": 8084,
            "pages": 252268,
            "totalEstimatedMatches": 252268,
            "someResultsRemoved": 67975}
    fig, ax = plt.subplots()
    y_pos= np.arange(len(data))
    ax.barh(y_pos, list(data.values()), align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.keys())
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Count of non zero values')
    plt.show()


def non_zero_per_type():
    keys = ["about", "someResultsRemoved", "full_path", "infection", "deep_links", "part_path", "related_searches",
            "pages", "fresh", "totalEstimatedMatches"]
    malicious = [1258, 9235, 5791, 2471, 1780, 9712, 151, 52710, 1111, 52710]
    clean = [74285, 58769, 73845, 5615, 104570, 183376, 0, 199670, 94417, 199670]

    malicious.extend(clean)
    keys.extend(keys)
    kind = ["malicious" if i < 10 else "clean" for i in range(20)]
    df = pandas.DataFrame({"data": malicious, "features": keys, "label": kind})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="features", hue="label", y="data", data=df).set_title("Count of non zero features per label")
    plt.show()


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


def feature_pair_plot():
    cols = ['part_path', 'fresh', 'pages', 'totalEstimatedMatches', "label"]
    pp = sns.pairplot(data=dataset[cols],
                      hue='label',
                      size=1.8, aspect=1.8,
                      palette={1: "#FF0000", 0: "#008000"},
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
    plt.show()

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

def plot_linear_reg():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    X_new = np.array([[0], [2]])
    y_predict = lin_reg.predict(X_new)
    plt.plot(X_new, y_predict, "r-")
    plt.plot(X, y, "b.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([0, 2, 0, 15])
    plt.show()


def logistic_reg():
    trues = 2 * np.random.rand(100, 1)
    falses = 4 + 3 * trues + np.random.randn(100, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right')
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(5, color='black')
    plt.show()

def svm():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = [
              'LinearSVC (linear kernel)',

              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((lin_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 1, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

probability_graphs()
# total_est_mtach_hist()