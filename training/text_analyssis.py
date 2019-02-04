import pickle

import pandas
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from tensorflow import confusion_matrix
import matplotlib.pyplot as plt

dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)

# tvec = TfidfVectorizer(min_df=.0025, max_df=.1, stop_words='english')
# tvec_weights = tvec.fit_transform(dataset.text)
# weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
# weights_df = pandas.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
# weights_df.sort_values(by='weight', ascending=False).head(20)
# print(weights_df)

# tfidf = TfidfVectorizer(min_df=0.2, analyzer='word', stop_words="english", ngram_range=(1, 2))
# features = tfidf.fit_transform(dataset.text)
# features_names = tfidf.get_feature_names()
labels = dataset.label
# pickle.dump(tfidf, open("tfidf.pkl", "wb"))

# print(features.shape)
# dataset = None

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pandas.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=30):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        # D = np.asarray(Xtr)
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=30):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y == label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs, min_df):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    transaltion = {"0": "čisté", "1": "škodlivé"}
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i + 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Průměrné Tf-Idf Skore pro {}% minimum".format(min_df), labelpad=16, fontsize=14)
        ax.set_title(transaltion[str(df.label)], fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1] + 1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.savefig("tfidf_{}.png".format(min_df))
    # plt.show()


for i in range(1, 6):
    tfidf = pickle.load(open("splitted_text/tf_idf/tf_idf_{}.pkl".format(i), "rb"))
    features = tfidf.transform(dataset.text)
    features_names = tfidf.get_feature_names()
    top_feats = top_feats_by_class(features, labels, features_names)
    print(top_feats)
    plot_tfidf_classfeats_h(top_feats, i)
# print(top_feats_in_doc(features, features_names, 42))
# print(top_mean_feats(features,features_names))
# print(top_feats_by_class(features, labels, features_names))
# plot_tfidf_classfeats_h(top_feats_by_class(features, labels, features_names))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 0, test_size=0.2)
# # X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], random_state = 0)
# # tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', stop_words='english')
# # features = tfidf.fit_transform(X_train)
# # features_chi2 = chi2(features, y_train)
# # print(features_chi2)
# # count_vect = CountVectorizer()
# # X_train_counts = count_vect.fit_transform(X_train)
# # tfidf_transformer = TfidfTransformer()
# # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# # clf = LogisticRegression().fit(X_train_tfidf, y_train)
# clf = LogisticRegression().fit(X_train, y_train)
#
# predictions = clf.predict(tfidf.transform(X_test))
# print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))

# import csv
#
# domains = set()
# with open("test_data2.csv", "r", encoding="utf-8") as file:
#     reader = csv.reader(file)
#     next(reader)
#     for row in reader:
#         domains.add(row[12])
# with open("text_test_data.csv", "r", encoding="utf-8") as file:
#     with open("text_test_data2.csv", "w", encoding="utf-8") as output:
#         reader = csv.reader(file, delimiter=";")
#         for row in reader:
#             if row[2] in domains:
#                 output.write("{};{};{}\n".format(row[0],row[1], row[2]))
#             else:
#                 print(row[2])