import pickle
import pandas
import time
import numpy as np
import os
import gensim
from scipy import  sparse
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import model_selection
# import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.feature_selection import f_regression
from sklearn.ensemble import BaggingClassifier

# dataset = pandas.read_csv("result.csv")
dataset = pandas.read_csv("test_data.csv", index_col=12)
dataset = dataset.sort_index()
print(dataset.shape)
print(round(dataset.describe(), 2))
print(dataset.groupby("label").size())
del dataset["ranking_response"]
# dataset.drop(['label'], axis=1).plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
# plt.show()
# dataset.drop(['label'], axis=1).hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

if not os.path.exists("binaries/x(0.3).npy") and not os.path.exists("binaries/y(0.3).npy"):
    if not os.path.exists("binaries/tfidf(0.1).pkl"):
        test_dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
        test_dataset = test_dataset.replace(np.nan, '', regex=True)
        test_dataset = test_dataset.sort_index()
        tfidf = TfidfVectorizer(min_df=0.2, analyzer='word', stop_words="english", ngram_range=(1, 2))
        features = tfidf.fit_transform(test_dataset.text)
        lda_model = gensim.models.LdaMulticore.load("gensim_lda/lda_model_20.pkl")
        bow_corpus = pickle.load(open("gensim_lda/bow_corpus.pkl", "rb"))
        topics = np.array([[doc_topics[0][0]] for doc_topics in lda_model.get_document_topics(bow_corpus)])
        lda_model, bow_corpus = None, None
        pickle.dump(tfidf, open("tfidf.pkl", "wb"))
    else:
        test_dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";",
                                       engine="python")
        test_dataset = test_dataset.replace(np.nan, '', regex=True)
        test_dataset = test_dataset.sort_index()
        tfidf = pickle.load(open("binaries/tfidf(0.1).pkl", "rb"))
        features = tfidf.transform(test_dataset.text)
        # features = pickle.load(open("gensim_fasttext/cluster_labels_tfidf.pkl", "rb")).reshape((-1,1))
        test_dataset, tfidf = None, None
        topics = pickle.load(open("gensim_lda/topics_20.pkl", "rb"))
        # w2v_labels = pickle.load(open("w2v_feature_array.pkl", "rb"))
        w2v_labels = pickle.load(open("cluster_labels.pkl", "rb")).reshape((-1,1))

    array = dataset.values
    X = array[:, 0:10]
    Y = array[:, 10]
    X = sparse.hstack([features, topics, w2v_labels, X])
    # X = np.concatenate((features, topics, w2v_labels, X), axis=1)
    # X = sparse.hstack([features, X])
    features, topics, array = None, None, None
    np.save("x.npy", X)
    np.save("y.npy", Y)
else:
    X = np.load("binaries/x(0.1).npy").tolist().toarray()
    Y = np.load("binaries/y(0.1).npy")
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation, indices_train, indices_test = model_selection.train_test_split(X, Y,
                                                                                                             dataset.index,
                                                                                                             test_size=validation_size,
                                                                                                             random_state=seed)
X, Y, = None, None
X_train = X_train.toarray()
scoring = 'accuracy'

print("Variances: {}".format(dataset.var()))
print("Correlations: {}".format(dataset.corr()))
# rf_exp = ExtraTreesClassifier(n_estimators=50)
# rf_exp = rf_exp.fit(X_train, Y_train)
# importances = list(rf_exp.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# [print('Variable(Extra treee classifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
# rf_exp = DecisionTreeClassifier()
# rf_exp = rf_exp.fit(X_train, Y_train)
# importances = list(rf_exp.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# [print('Variable(decision treee classifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
#
# rf_exp = GradientBoostingClassifier()
# rf_exp = rf_exp.fit(X_train, Y_train)
# importances = list(rf_exp.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# [print('Variable(GradientBoostingClassifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
# # rf_exp = RandomForestRegressor(n_estimators=1000, random_state=100)
# # rf_exp.fit(X_train, Y_train)
# # importances = list(rf_exp.feature_importances_)
# # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
# # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# # [print('Variable(Random forrest regressor) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#
# model = SelectFromModel(rf_exp, prefit=True)
# X_new = model.transform(X_train)
# print("Select from model shape: {}".format(X_new.shape))
#
# model = SelectKBest(chi2, k=5)
# fit = model.fit(X_train, Y_train)
# print("Select k best priorities: {}".format(fit.scores_))
# X_new=fit.transform(X_train)
# print("Select k best with chi2: {}".format(X_new.shape))
#
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,max_iter=2000).fit(X, Y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X)
# print("Linear SVC: {}".format(X_new.shape))
# print(model.get_support())
#
# estimator = LogisticRegression(solver='lbfgs')
# selector = RFE(estimator, 5)
# selector = selector.fit(X, Y)
# print("RFE count features: {}, features: {}, ranking: {}".format(selector.n_features_, selector.support_, selector.ranking_))
#
# pca = decomposition.PCA(n_components=12)
# fit = pca.fit(X_train, Y_train)
# X_new = pca.transform(X_train)
# print("PCA: {}".format(X_new.shape))
# # print("PCA components: {}".format(fit.components_))
# # print("PCA variance: {}".format(fit.explained_variance_ratio_))
#
# ica=FastICA(n_components=12, random_state=0)
# x_reduced=ica.fit_transform(X_train)
# print("ICA: {}".format(x_reduced.shape))

models = []
# from sklearn.preprocessing import MinMaxScaler
# scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_validation = scaling.transform(X_validation.toarray())

models.append(('LR', LogisticRegression(solver='lbfgs', class_weight="balanced")))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecTree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('RForest', RandomForestClassifier(n_estimators=100)))
models.append(('Kmeans', KMeans()))
models.append(('Ada', AdaBoostClassifier()))
# models.append(('SVM(linear)', LinearSVC(max_iter=2000)))
# models.append(('SVC', SVC(gamma="scale", cache_size=1000)))
# models.append(('NuSVC', NuSVC(gamma="scale", cache_size=1000)))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('SQD', SGDClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('CatBoost', CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')))
models.append(('BNB', BernoulliNB()))
models.append(('RC', RidgeClassifier()))
models.append(('perc', Perceptron()))
models.append(('passive', PassiveAggressiveClassifier()))
models.append(('nearest', NearestCentroid()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))

results = []
names = []
start_time = time.time()
for name, model in models:
    try:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    except Exception as e:
        print(e)
    else:
        print(msg)
print("Took: {}".format(time.time() - start_time))

# # train_data = lgb.Dataset(X_train, label=Y_train)
# lgb_train = lgb.Dataset(X_train, Y_train)
# lgb_eval = lgb.Dataset(X_validation, Y_validation, reference=lgb_train)
# # param = {'num_leaves': 31, 'num_trees': 50, 'objective': 'binary', 'metric': 'auc'}
# param = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'l2', 'auc'},
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
# bst = lgb.train(param, lgb_train, 10, valid_sets=lgb_eval)
# y_pred = bst.predict(X_validation, num_iteration=bst.best_iteration)
# print('The mean of prediction is:', y_pred.mean(), y_pred.std())


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation.toarray())
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
# for input, prediction, label in zip(indices_test, predictions, Y_validation):
#     if prediction != label:
#         print("Domain {} with incorrect label: {}, should be: {}".format(input, prediction, label))

# fig = plt.figure(figsize=(10.0, 8.0))
# ax = fig.add_subplot(111)
# plt.boxplot(results, 0, '')
# ax.set_xticklabels(names)
# plt.ylabel("Přesnost")
# for tick in ax.get_xticklabels():
#     tick.set_rotation(70)
# plt.savefig("test_data.png")
with open("test/minimal_test.txt", "w") as file:
    for name, result in zip(names, results):
        file.write("{},{}\n".format(name, result))
# plt.show()
hyper_parameters = {
    "Ada": {
        "n_estimators ": [50, 100, 200, 300, 400, 500],
        "algorithm ": ["SAMME", "SAMME.R"],
        "learning_rate ": [0.2, 0.3, 0.5, 0.75, 1.0, 2, 5]
    },
    'LR': {
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "class_weight": ["balanced"],
        "penalty": ["l1", "l2"],
        "intercept_scaling ": [0.1, 0.5, 1, 5, 10],
        "C": [0.5, 0.3, 0.1, 1, 0.8, 0.6, 3, 5, 10],
        "max_iter": [100, 200, 300, 400, 500],
        "n_jobs": -1
    },
    'LDA': {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": [None, "auto", 0.1, 0.5, 1],
        "n_components": [4, 6, 8, 12]
    },
    "KNN": {
        "n_neighbors": [3, 5, 8, 10, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
        "leaf_size": [5, 10, 20, 30, 50, 60],
        "p": [1, 2, 3, 4, 5],
        "n_jobs": -1
    },
    "DecTree": {
        "max_depth": [None, 8, 12, 16, 32],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": ["balanced", None],
        "min_impurity_decrease": [0, 0.1, 0.2, 0.3]
    },
    "MNB": {
        "alpha": [0.01, 0.5, 1, 2, 5],
        "fit_prior": [True, False]
    },
    "RForest": {
        "n_estimators": [100, 200, 300, 500, 800],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 8, 12, 16, 32],
        "max_features": [None, "sqrt", "log2"],
        "min_impurity_decrease": [0.1, 0.2, 0.3, 0.5],
        "oob_score": [False, True],
        "class_weight": ["balanced", None, "balanced_subsample"],
        "n_jobs": -1
    },
    "Kmeans": {
        "n_clusters": [5, 8, 10, 15, 25],
        "init": ["kmeans++", "random"],
        "n_init": [10, 15, 20, 25, 30],
        "max_iter ": [200, 300, 500, 600],
        "precompute_distances": ["auto", True, False],
        "algorithm": ["auto", "full", "elkan"],
        "n_jobs": -1
    },
    "GBC": {
        "learning_rate ": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
        "loss": ["deviance", "exponential"],
        "n_estimators ": [100, 200, 300, 400, 500],
        "subsample": [0.2, 0.7, 1.0, 1.5, 2.0],
        "criterion": ["friedman_mse", "mse", "mae"],
        "max_depth": [3, 8, 12, 16, 32],
        "min_impurity_decrease": [0, 0.1, 0.2, 0.3],
        "max_features": [None, "sqrt", "log2"],
    },
    "SQD": {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": [0.001, 0.1, 0.2, 0.4, 0.5, 0.8],
        "max_iter": [500, 1000, 1500, 2000, 3000],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [0.001, 0.1, 0.2, 0.4, 0.5, 0.8],
        "class_weight": ["balanced", None],
        "average": [True, 10, 100, 1000],
        "n_jobs": -1
    },
    "XGB": {
        "eta": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
        "gamma": [0, 0.2, 0.8, 5, 10, 20],
        "max_depth": [0, 6, 10, 25, 30],
        "subsample": [0.2, 0.5, 0.75, 1],
        "tree_method": ["auto", "exact", "approx", "hist"],
        "updater": ["grow_colmaker,prune", "grow_histmaker,sync", ],
        "process_type": ["default", "update"],
        "grow_policy": ["depthwise", "lossguide"],
        "num_parallel_tree": [1, 3, 5, 10, 15],
        "max_bin": [256, 512, 1024]
    },
    "CatBoost": {
        "loss_function": ["Logloss", "CrossEntropy", "LogLinQuantile", "Quantile"],
        'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
        'iterations': [50, 250, 100, 500, 1000],
        'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
        'l2_leaf_reg': [3, 1, 5, 10, 100],
        'border_count': [32, 5, 10, 20, 50, 100, 200],
        'ctr_border_count': [50, 5, 10, 20, 100, 200],
        "use_best_model": True,
        "eval_metric": "Accuracy",
        "thread_count": -1
    },
    "BNB": {
        "alpha": [0.01, 0.5, 1, 2, 5],
        "fit_prior": [True, False]
    },
    "RC": {
        "alpha": [0.5, 0.1, 0.4, 0.2, 1, ],
        "normalize": [True, False],
        "max_iter": [50, 250, 100, 500],
        "class_weight": "balanced",
        "solver": ["svd", "sparse_cg", "lsqr", "sag"],
    },
    "perc": {
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": [0.5, 0.1, 0.4, 0.2, 1],
        "max_iter": [500, 1000, 1500, 2000, 3000],
        "tol": 1e-3,
        "early_stopping": True,
        "class_weight": ["balanced", None],
        "n_jobs": -1
    },
    "passive": {
        "C": [0.5, 0.3, 0.1, 0.2, 1, 0.8, 0.6],
        "max_iter": [500, 1000, 1500, 2000, 3000],
        "tol": 1e-3,
        "early_stopping": True,
        "loss": ["hinge", "squared_hinge"],
        "class_weight": ["balanced", None],
        "n_jobs": -1
    },
    "QDA": {
        "tol": 1e-4,
        "store_covariance": True
    },
    "SVM(linear)": {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [False, True],
        "C": [0.5, 0.2, 1, 0.8, 0.6, 2, 3, 5],
        "intercept_scaling": [1, 2, 3, 5, 10],
        "random_state": seed,
        "max_iter": [1000, 2000, 3000, 4000]
    }
}
