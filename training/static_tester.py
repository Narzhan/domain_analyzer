import pickle
import pandas
import time
import numpy as np
import os
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import gensim
from scipy import  sparse
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import model_selection
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, LinearRegression
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
# dataset = pandas.read_csv("test_data.csv", index_col=12)
# dataset = dataset.sort_index()
# dataset_lda = pandas.read_csv("splitted_text/lda/result_data.csv", index_col=0)
# dataset = dataset.join(dataset_lda)
# dataset_tfidf = pandas.read_csv("splitted_text/tf_idf/result_data.csv", index_col=0)
# dataset = dataset.join(dataset_tfidf)
# dataset_we = pandas.read_csv("splitted_text/word_embedding/result_data.csv", index_col=0)
# dataset = dataset.join(dataset_we)
# col_list = list(dataset)
# col_list.insert(len(col_list), col_list.pop(col_list.index('label')))
# dataset = dataset.ix[:, col_list]

dataset = pandas.read_csv("dataframe_enhanced.csv", index_col=0,
                          usecols=['part_path', 'fresh', 'pages', 'totalEstimatedMatches', "topics", "tf_idf",
                                   # "embedding",
                                   'label', "domain"]
                          )
# del dataset["embedding"]
print(dataset.shape)
print(round(dataset.describe(), 2))
print(dataset.groupby("label").size())
# del dataset["ranking_response"]
# dataset.to_csv("dataframe_enhanced.csv")
# dataset.drop(['label'], axis=1).plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
# plt.show()
# dataset.drop(['label'], axis=1).hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()

if not os.path.exists("binaries/x(0.3).npy") and not os.path.exists("binaries/y(0.3).npy"):
    # if not os.path.exists("binaries/tfidf(0.1).pkl"):
    #     test_dataset = pandas.read_csv("text_test_data.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
    #     test_dataset = test_dataset.replace(np.nan, '', regex=True)
    #     test_dataset = test_dataset.sort_index()
    #     tfidf = TfidfVectorizer(min_df=0.2, analyzer='word', stop_words="english", ngram_range=(1, 2))
    #     features = tfidf.fit_transform(test_dataset.text)
    #     lda_model = gensim.models.LdaMulticore.load("gensim_lda/lda_model_20.pkl")
    #     bow_corpus = pickle.load(open("gensim_lda/bow_corpus.pkl", "rb"))
    #     topics = np.array([[doc_topics[0][0]] for doc_topics in lda_model.get_document_topics(bow_corpus)])
    #     lda_model, bow_corpus = None, None
    #     pickle.dump(tfidf, open("tfidf.pkl", "wb"))
    # else:
    #     # test_dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";",
    #     #                                engine="python")
    #     # test_dataset = test_dataset.replace(np.nan, '', regex=True)
    #     # test_dataset = test_dataset.sort_index()
    #     # tfidf = pickle.load(open("binaries/tfidf(0.1).pkl", "rb"))
    #     # features = tfidf.transform(test_dataset.text)
    #     features = pickle.load(open("gensim_fasttext/cluster_labels_tfidf.pkl", "rb")).reshape((-1,1))
    #     test_dataset, tfidf = None, None
    #     topics = pickle.load(open("gensim_lda/topics_20.pkl", "rb"))
    #     # w2v_labels = pickle.load(open("w2v_feature_array.pkl", "rb"))
    #     w2v_labels = pickle.load(open("cluster_labels.pkl", "rb")).reshape((-1,1))
    array = dataset.values
    X = array[:, 0:-1]
    Y = array[:, -1]
    # X = sparse.hstack([features, topics, w2v_labels, X])
    # X = np.concatenate((features, topics, w2v_labels, X), axis=1)
    # X = sparse.hstack([features, X])
    # features, topics, array = None, None, None
    # np.save("x.npy", X)
    # np.save("y.npy", Y)
else:
    X = np.load("binaries/x(0.1).npy").tolist().toarray()
    Y = np.load("binaries/y(0.1).npy")
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation, indices_train, indices_test = model_selection.train_test_split(X, Y,
                                                                                                             dataset.index,
                                                                                                             test_size=validation_size,
                                                                                                             random_state=seed,
                                                                                                             stratify=Y
                                                                                                             )
X, Y, = None, None
# X_train = X_train.toarray()
scoring = 'accuracy'

print("Variances: {}".format(dataset.var()))
print("Correlations: {}".format(dataset.corr()))
# knn = KNeighborsClassifier()
# knn = RandomForestClassifier(n_estimators=100)
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
# with open("random_forest.txt", "a") as file:
#     file.write(str(accuracy_score(Y_validation, predictions)))
#     file.write(str(confusion_matrix(Y_validation, predictions)))
#     file.write(classification_report(Y_validation, predictions))
# raise Exception

# from sklearn.preprocessing import MinMaxScaler
# scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_validation = scaling.transform(X_validation)
#
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
# #C=0.01, penalty="l1", dual=False, max_iter=2000
# lsvc = LinearSVC().fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X_train)
# print("Linear SVC: {}".format(X_new.shape))
# print("LinearSVC coefs {}".format(lsvc.coef_))
# print(model.get_support())
#
# lsvc = LogisticRegression(solver='lbfgs', class_weight="balanced").fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X_train)
# print("LogReg: {}".format(X_new.shape))
# print(model.get_support())
#
# lsvc = lgb.LGBMClassifier(objective= 'binary').fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X_train)
# print("Light: {}".format(X_new.shape))
# print(model.get_support())
#
# lsvc = GradientBoostingClassifier().fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X_train)
# print("GBc: {}".format(X_new.shape))
# print(model.get_support())
#
# lsvc = RandomForestClassifier(n_estimators=100).fit(X_train, Y_train)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X_train)
# print("RandForest: {}".format(X_new.shape))
# print(model.get_support())
#
# estimator = LogisticRegression(solver='lbfgs')
# selector = RFE(estimator, 5)
# selector = selector.fit(X_train, Y_train)
# print("RFE count features: {}, features: {}, ranking: {}".format(selector.n_features_, selector.support_, selector.ranking_))
#
# pca = decomposition.PCA(n_components=7)
# fit = pca.fit(X_train, Y_train)
# X_new = pca.transform(X_train)
# print("PCA: {}".format(X_new.shape))
# print("PCA components: {}".format(pca.n_components_))
# print("PCA variance: {}".format(pca.explained_variance_ratio_))
#
# ica=FastICA(n_components=11, random_state=0)
# x_reduced=ica.fit_transform(X_train)
# print("ICA: {}".format(x_reduced.shape))
# # print("ICA components: {}".format(ica.components_))
# # print("ICA mixing: {}".format(ica.mixing_ ))subl

models = []
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer

# for name, scaler in {"standard": StandardScaler
#                      # "max": MaxAbsScaler, "robust": RobustScaler,
#                      # "quantile": QuantileTransformer, "power": PowerTransformer
#                      }.items():
#     scaling = scaler().fit(X_train)
#     X_scaled = scaling.transform(X_train)
#     pickle.dump(scaling, open("pca/{}.pkl".format(name), "wb"))
#
#     pca = decomposition.PCA(n_components=5)
#     X_train = pca.fit_transform(X_scaled, Y_train)
#     pickle.dump(pca, open("pca/pca.pkl", "wb"))
#
#     for model_name, model in {"knn": KNeighborsClassifier(), "forests": RandomForestClassifier(n_estimators=100),
#                               "lsvc": LinearSVC(max_iter=2000)}.items():
#         model.fit(X_train, Y_train)
#         pickle.dump(model, open("pca/{}_model_{}".format(name, model_name), "wb"))
#     param = {"boosting_type": "gbdt",
#              "objective": "binary"}
#     light = lgb.LGBMClassifier(**param)
#     light.fit(X_train, Y_train)
#     pickle.dump(light, open("pca/{}_model_lightgbm.pkl".format(name), "wb"))

scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_validation = scaling.transform(X_validation)
# pickle.dump(scaling, open("binaries/scaler.pkl", "wb"))


models.append(('LR', LogisticRegression(solver='lbfgs', class_weight="balanced")))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecTree', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('RForest', RandomForestClassifier(n_estimators=100)))
models.append(('Kmeans', KMeans()))
models.append(('Ada', AdaBoostClassifier()))
models.append(('SVM(linear)', LinearSVC(max_iter=2000)))
# models.append(('SVC', SVC(gamma="scale", cache_size=1000)))
# models.append(('NuSVC', NuSVC(gamma="scale", cache_size=1000)))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('SQD', SGDClassifier(max_iter=1000, tol=1e-3)))
models.append(('XGB', XGBClassifier()))
models.append(('CatBoost', CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')))
models.append(('BNB', BernoulliNB()))
models.append(('RC', RidgeClassifier()))
models.append(('perc', Perceptron(max_iter=1000, tol=1e-3)))
models.append(('passive', PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)))
models.append(('nearest', NearestCentroid()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))


results = []
names = []
start_time = time.time()
for name, model in models:
    try:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    except Exception as e:
        print(e)
    else:
        print(msg)
print("Took: {}".format(time.time() - start_time))
#
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_validation, Y_validation, reference=lgb_train)
# param = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'l2', 'auc', "binary_logloss"},
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
param = {"boosting_type": "gbdt", "colsample_bytree": 1, "min_child_samples": 300, "min_child_weight": 0.001, "num_leaves": 40, "objective": "binary", "reg_alpha": 2, "reg_lambda": 0.5, "subsample": 0.5, "verbose": 0}
bst = lgb.train(param, lgb_train, 20, valid_sets=lgb_eval)
y_pred = bst.predict(X_validation, num_iteration=bst.best_iteration)
for i in range(0, Y_validation.shape[0]):
    if y_pred[i] >= .5:  # setting threshold to .5
        y_pred[i] = 1
    else:
        y_pred[i] = 0
cm = confusion_matrix(Y_validation, y_pred)
print(cm)
accuracy = accuracy_score(y_pred, Y_validation)
print("LightGBM: {}".format(accuracy))

# param = {"boosting_type": "gbdt", "colsample_bytree": 1, "min_child_samples": 300, "min_child_weight": 0.001,
#          "num_leaves": 40, "objective": "binary", "reg_alpha": 2, "reg_lambda": 0.5, "subsample": 0.5, "verbose": 0}
# light = lgb.LGBMClassifier(**param)
# light.fit(X_train, Y_train)
# pickle.dump(light, open("binaries/lightgbm.pkl", "wb"))
# # predictions = light.predict(X_validation)
# # for i in range(0, Y_validation.shape[0]):
# #     if predictions[i] >= .5:  # setting threshold to .5
# #         predictions[i] = 1
# #     else:
# #         predictions[i] = 0
# # with open("mistakes/fp_fn_lightgbm.txt", "w") as file:
# #     counter = 0
# #     for input, prediction, label in zip(indices_test, predictions, Y_validation):
# #         if prediction != label:
# #             file.write("Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
# #                                                                                               list(X_validation[
# #                                                                                                        counter])))
# #         counter += 1
# param = {"C": 0.2, "dual": False, "intercept_scaling": 2, "loss": "squared_hinge", "max_iter": 2500, "penalty": "l2"}
# lsvc = LinearSVC(**param)
# lsvc.fit(X_train, Y_train)
# pickle.dump(lsvc, open("binaries/linearsvc.pkl", "wb"))
# # predictions = lsvc.predict(X_validation)
# # with open("mistakes/fp_fn_svc.txt", "w") as file:
# #     counter = 0
# #     for input, prediction, label in zip(indices_test, predictions, Y_validation):
# #         if prediction != label:
# #             file.write("Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
# #                                                                                               list(X_validation[
# #                                                                                                        counter])))
# #         counter += 1
# param = {"class_weight": None, "criterion": "entropy", "max_depth": 9, "max_features": None, "min_impurity_decrease": 0,
#          "min_samples_leaf": 4, "min_samples_split": 13, "n_estimators": 700, "oob_score": False}
# rforrest = RandomForestClassifier(**param)
# rforrest.fit(X_train, Y_train)
# pickle.dump(rforrest, open("binaries/rforest.pkl", "wb"))
# # predictions = rforrest.predict(X_validation)
# # with open("mistakes/fp_fn_rforrest.txt", "w") as file:
# #     counter = 0
# #     for input, prediction, label in zip(indices_test, predictions, Y_validation):
# #         if prediction != label:
# #             file.write("Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
# #                                                                                               list(X_validation[
# #                                                                                                        counter])))
# #         counter += 1
# param = {'algorithm': 'kd_tree', 'leaf_size': 20, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# knn = KNeighborsClassifier(**param)
# knn.fit(X_train, Y_train)
# pickle.dump(knn, open("binaries/knn.pkl", "wb"))
# # predictions = knn.predict(X_validation)
# # print(accuracy_score(Y_validation, predictions))
# # print(confusion_matrix(Y_validation, predictions))
# # print(classification_report(Y_validation, predictions))
# # with open("mistakes/fp_fn_knn.txt", "w") as file:
# #     counter = 0
# #     for input, prediction, label in zip(indices_test, predictions, Y_validation):
# #         if prediction != label:
# #             file.write("Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
# #                                                                                               list(X_validation[
# #                                                                                                        counter])))
# #         counter += 1
#
# rforrest = XGBClassifier()
# rforrest.fit(X_train, Y_train)
# pickle.dump(rforrest, open("binaries/xgb.pkl", "wb"))
#
# rforrest = BernoulliNB()
# rforrest.fit(X_train, Y_train)
# pickle.dump(rforrest, open("binaries/bnb.pkl", "wb"))
#
# rforrest = RidgeClassifier()
# rforrest.fit(X_train, Y_train)
# pickle.dump(rforrest, open("binaries/rc.pkl", "wb"))

# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# fig = plt.figure(figsize=(10.0, 8.0))
# ax = fig.add_subplot(111)
# plt.boxplot(results, 0, '')
# ax.set_xticklabels(names)
# plt.ylabel("Přesnost")
# for tick in ax.get_xticklabels():
#     tick.set_rotation(70)
# plt.savefig("test_data.png")
# with open("test/minimal_test.txt", "w") as file:
#     for name, result in zip(names, results):
#         file.write("{},{}\n".format(name, result))
# plt.show()
# hyper_parameters = {
#     "Ada": {
#         "n_estimators": [50, 100, 200, 300, 400, 500],
#         "algorithm": ["SAMME", "SAMME.R"],
#         "learning_rate": [0.2, 0.3, 0.5, 0.75, 1.0, 2, 5]
#     },
#     'LR': {
#         "solver": ["newton-cg", "lbfgs"],
#         "class_weight": ["balanced"],
#         "penalty": ["l2"],
#         "intercept_scaling": [0.1, 0.5, 1, 5, 10],
#         "C": [1, 0.8, 1.5],
#         "max_iter": [400, 500, 600]
#     },
#     'LDA': {
#         "solver": ["svd", "lsqr", "eigen"],
#         "n_components": [2, 4, 6, 8, 12, 20, 30]
#     },
#     "KNN": {
#         "n_neighbors": [3, 5, 8, 10, 15],
#         "weights": ["uniform", "distance"],
#         "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
#         "leaf_size": [5, 10, 20, 30, 50, 60],
#         "p": [1, 2, 3, 4, 5]
#     },
#     "DecTree": {
#         "max_depth": [None, 8, 12, 16, 32],
#         "criterion": ["gini", "entropy"],
#         "splitter": ["best", "random"],
#         "max_features": [None, "sqrt", "log2"],
#         "class_weight": ["balanced", None],
#         "min_impurity_decrease": [0, 0.1, 0.2, 0.3]
#     },
#     "MNB": {
#         "alpha": [0.01, 0.5, 1, 2, 5],
#         "fit_prior": [True, False]
#     },
#     "RForest": {
#         "n_estimators": [x *100 for x in range(1, 11)],
#         "criterion": ["gini", "entropy"],
#         "max_depth": [None, 8, 12, 16, 32],
#         "max_features": [None, "sqrt", "log2"],
#         "min_samples_split": [2, 5, 10, 15, 20],
#         "min_samples_leaf": [1, 2, 4, 6, 8],
#         "min_impurity_decrease": [0, 0.1, 0.2, 0.3, 0.5],
#         "oob_score": [False, True],
#         "class_weight": ["balanced", None, "balanced_subsample"]
#     },
#     "GBC": {
#             "learning_rate": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
#             "loss": ["deviance", "exponential"],
#             "n_estimators": [x * 100 for x in range(1, 7)],
#             "subsample": [0.1, 0.2, 0.4, 0.6, 0.7, 0.8],
#             "criterion": ["friedman_mse", "mse", "mae"],
#             "max_depth": [3, 8, 12, 16, 32],
#             "min_samples_split": [0.1, 0.3, 0.5, 0.8, 1.0],
#             "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 1],
#             "min_impurity_decrease": [0, 0.1, 0.2, 0.3],
#             "max_features": [None, "sqrt", "log2"]
#     },
#     "SQD": {
#         "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
#         "penalty": ["l1", "l2", "elasticnet"],
#         "alpha": [0.001, 0.1, 0.2, 0.4, 0.5, 0.8],
#         "max_iter": [500, 1000, 1500, 2000, 3000],
#         "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
#         "eta0": [0.001, 0.1, 0.2, 0.4, 0.5, 0.8],
#         "class_weight": ["balanced", None],
#         "average": [True, 10, 100, 1000],
#     },
#     "XGB": {
#         "eta": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
#         "gamma": [0, 0.2, 0.8, 5, 10, 20],
#         "max_depth": [0, 6, 10, 25, 30],
#         "subsample": [0.2, 0.5, 0.75, 1],
#         "tree_method": ["auto", "exact", "approx", "hist"],
#         "updater": ["grow_colmaker,prune", "grow_histmaker,sync", ],
#         "process_type": ["default", "update"],
#         "grow_policy": ["depthwise", "lossguide"],
#         "num_parallel_tree": [1, 3, 5, 10, 15],
#         "max_bin": [256, 512, 1024]
#     },
#     "CatBoost": {
#         "loss_function": ["Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll"],
#         'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
#         'iterations': [50, 250, 100, 500, 1000],
#         'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
#         'l2_leaf_reg': [3, 1, 5, 10, 100],
#         'border_count': [32, 5, 10, 20, 50, 100, 200],
#         'ctr_border_count': [50, 5, 10, 20, 100, 200],
#         "logging_level": ["Silent"],
#         "eval_metric": ["Accuracy"]
#     },
#     "BNB": {
#         "alpha": [0.01, 0.5, 1, 2, 5],
#         "fit_prior": [True, False]
#     },
#     "RC": {
#         "alpha": [0.5, 0.1, 0.4, 0.2, 1, ],
#         "normalize": [True, False],
#         "max_iter": [50, 250, 100, 500],
#         "class_weight": ["balanced"],
#         "solver": ["svd", "sparse_cg", "lsqr", "sag"],
#     },
#     "perc": {
#         "penalty": ["l1", "l2", "elasticnet"],
#         "alpha": [0.5, 0.1, 0.4, 0.2, 1],
#         "max_iter": [500, 1000, 1500, 2000, 3000],
#         "tol": [1e-3],
#         "early_stopping": [True],
#         "class_weight": ["balanced", None]
#     },
#     "passive": {
#         "C": [0.5, 0.3, 0.1, 0.2, 1, 0.8, 0.6],
#         "max_iter": [500, 1000, 1500, 2000, 3000],
#         "tol": [1e-3],
#         "early_stopping": [True],
#         "loss": ["hinge", "squared_hinge"],
#         "class_weight": ["balanced", None]
#     },
#     "SVM(linear)": {
#         "penalty": ["l1", "l2"],
#         "loss": ["hinge", "squared_hinge"],
#         "dual": [False, True],
#         "C": [0.5, 0.2, 1, 0.8, 0.6, 2, 3, 5],
#         "intercept_scaling": [1, 2, 3, 5, 10],
#         "max_iter": [1000, 2000, 3000, 4000]
#     },
#     "LightGBM": {
#         'objective': ['binary'],
#         'boosting_type': ['gbdt', "dart"],
#         'num_leaves': sp_randint(6, 50),
#         'min_child_samples': sp_randint(100, 500),
#         'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#         'subsample': sp_uniform(loc=0.2, scale=0.8),
#         'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
#         'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#         'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
#     }
# }

grid_hyper_parameters = {
    "Ada": {
        "n_estimators": [300, 400, 500],
        "algorithm": ["SAMME.R"],
        "learning_rate": [0.2, 0.3, 0.4]
    },
    'LR': {
        "solver": ["newton-cg", "lbfgs"],
        "class_weight": ["balanced"],
        "penalty": ["l2"],
        "intercept_scaling": [3, 5, 10],
        "C": [0.9, 1, 0.8, 0.6],
        "max_iter": [400, 500, 600]
    },
    "DecTree": {
        "max_depth": [6, 7, 8, 9],
        "criterion": ["entropy"],
        "splitter": ["random"],
        "max_features": [None],
        "class_weight": [None],
        "min_impurity_decrease": [0, 0.1]
    },
    "KNN": {
        "n_neighbors": [3, 5, 6],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2]
    },
    "RForest": {
        "n_estimators": [500, 600, 700],
        "criterion": ["entropy"],
        "max_depth": [6, 8, 9],
        "max_features": [None],
        "min_samples_split": [13, 15, 18],
        "min_samples_leaf": [4, 5],
        "min_impurity_decrease": [0, 0.1],
        "oob_score": [False],
        "class_weight": [None]
    },
    "GBC": {
        "learning_rate": [0.1, 0.2, 0.3],
        "loss": ["deviance"],
        "n_estimators": [200, 300],
        "subsample": [0.7, 0.8],
        "criterion": ["friedman_mse", "mse"],
        "max_depth": [16, 32],
        "min_samples_split": [0.2, 0.3],
        "min_samples_leaf": [0.2, 0.3],
        "min_impurity_decrease": [0, 0.1],
        "max_features": ["sqrt", "log2"]
    },
    "SQD": {
        "loss": ["hinge", "perceptron"],
        "penalty": ["l2"],
        "alpha": [0.4, 0.5],
        "max_iter": [2000, 3000],
        "learning_rate": ["optimal", "invscaling"],
        "eta0": [0.1, 0.2],
        "class_weight": [None],
        "average": [10, 15],
    },
    "XGB": {
        "eta": [0.8, 1.0],
        "gamma": [0.8, 1],
        "max_depth": [6, 8],
        "subsample": [0.75, 0.85],
        "tree_method": ["auto", "exact"],
        "updater": ["grow_histmaker,sync"],
        "process_type": ["default"],
        "grow_policy": ["lossguide"],
        "num_parallel_tree": [3, 4],
        "max_bin": [1024]
    },
    "CatBoost": {
        "loss_function": ["MultiClass"],
        'depth': [9, 10, 12],
        'iterations': [500, 600],
        'learning_rate': [0.2, 0.3],
        'l2_leaf_reg': [100, 130],
        'border_count': [150, 200, 250],
        'ctr_border_count': [5, 6, 8],
        "eval_metric": ["Accuracy"],
        "verbose": [False]
    },
    "passive": {
        "C": [0.3, 0.1, 0.2],
        "max_iter": [1000, 1500, 2000],
        "tol": [1e-3, 0.02, 0.03],
        "early_stopping": [True],
        "loss": ["squared_hinge"],
        "class_weight": ["balanced"]
    },
    "SVM(linear)": {
        "penalty": ["l2"],
        "loss": ["squared_hinge"],
        "dual": [False, True],
        "C": [0.3, 0.2, 0.1],
        "intercept_scaling": [2, 3, 4],
        "max_iter": [2500, 3000, 3500]
    },
    "LightGBM": {
        'objective': ['binary'],
        'boosting_type': ['gbdt'],
        'num_leaves': [40, 50, 60],
        'min_child_samples': [300, 350, 400],
        'min_child_weight': [1e-3, 1e-2, 1e-1],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [0.8, 1],
        'reg_alpha': [1, 2, 5],
        'reg_lambda': [0, 1e-1, 0.5],
        "verbose": [0]
    }
}
# classifiers = {
# # 'LR': LogisticRegression(),
# 'KNN': KNeighborsClassifier()
# # 'SQD': SGDClassifier(),
# # 'LightGBM': lgb.LGBMClassifier(), 'passive': PassiveAggressiveClassifier(), 'RC': RidgeClassifier(),
# # 'LDA': LinearDiscriminantAnalysis(), 'perc': Perceptron(), 'MNB': MultinomialNB(),
# # 'CatBoost': CatBoostClassifier(), 'SVM(linear)': LinearSVC(), 'XGB': XGBClassifier(),
# # 'DecTree': DecisionTreeClassifier(), 'RForest': RandomForestClassifier(), 'BNB': BernoulliNB(),
# # 'GBC': GradientBoostingClassifier(), 'Ada': AdaBoostClassifier()
# }

# grid_classifiers = {
#     'LR': LogisticRegression(), 'DecTree': DecisionTreeClassifier(), 'RForest': RandomForestClassifier(),
#     'Ada': AdaBoostClassifier(), 'LightGBM': lgb.LGBMClassifier(), 'CatBoost': CatBoostClassifier(),
#     'SVM(linear)': LinearSVC()
#     # 'SQD': SGDClassifier(),
#     # 'passive': PassiveAggressiveClassifier(),
#     # 'XGB': XGBClassifier(),
#     # 'KNN': KNeighborsClassifier(), 'GBC': GradientBoostingClassifier()
# }
#
#
# for name, classifier in grid_classifiers.items():
#     print(name)
#     print("###############")
#     # rf_random = RandomizedSearchCV(estimator=classifier, param_distributions=grid_hyper_parameters[name], n_iter=100, cv=2,
#     #                                verbose=2, error_score=0.0,
#     #                                random_state=7, n_jobs=-1, scoring="accuracy")
#     rf_random = GridSearchCV(estimator=classifier, param_grid=grid_hyper_parameters[name], cv=2,
#                              verbose=2, error_score=0.0, n_jobs=-1, scoring="accuracy")
#     rf_random.fit(X_train, Y_train)
#     print(rf_random.best_params_)
#     best_random = rf_random.best_estimator_
#     predictions = best_random.predict(X_validation)
#     print(accuracy_score(Y_validation, predictions))
#     print(classification_report(Y_validation, predictions))
#     with open("{}.txt".format(name), "a") as file:
#         file.write(rf_random.best_params_)
#         file.write(str(accuracy_score(Y_validation, predictions)))
#         file.write(str(confusion_matrix(Y_validation, predictions)))
#         file.write(classification_report(Y_validation, predictions))


# results = []
# configs= []
# import random
# for _ in range(100):
#     parameters = {}
#     for parameter, values in grid_hyper_parameters["GBC"].items():
#         parameters[parameter] = random.choice(values)
#     try:
#         kfold = model_selection.KFold(n_splits=2, random_state=seed, shuffle=True)
#         cv_results = model_selection.cross_val_score(GradientBoostingClassifier(**parameters), X_train, Y_train, cv=kfold, scoring=scoring)
#         results.append(cv_results.mean())
#         configs.append(parameters)
#         msg = "%s: %f (%f)" % (parameters, cv_results.mean(), cv_results.std())
#     except Exception as e:
#         print(e)
#     else:
#         print(msg)
# index = results.index(max(results))
# print(configs[index])
# print("Best parameters: {}".format(configs[index]))