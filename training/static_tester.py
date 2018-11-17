import pandas
import time
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import model_selection
# import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
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
dataset = pandas.read_csv("test_data.csv", index_col=13)
print(dataset.shape)
print(round(dataset.describe(), 2))
print(dataset.groupby("label").size())
# dataset.drop(['label'], axis=1).plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
# plt.show()
# dataset.drop(['label'], axis=1).hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()


array = dataset.values
X = array[:, 0:12]
Y = array[:, 12]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

scoring = 'accuracy'

print("Variances: {}".format(dataset.var()))
print("Correlations: {}".format(dataset.corr()))
rf_exp = ExtraTreesClassifier(n_estimators=50)
rf_exp = rf_exp.fit(X_train, Y_train)
importances = list(rf_exp.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable(Extra treee classifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

rf_exp = DecisionTreeClassifier()
rf_exp = rf_exp.fit(X_train, Y_train)
importances = list(rf_exp.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable(decision treee classifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


rf_exp = GradientBoostingClassifier()
rf_exp = rf_exp.fit(X_train, Y_train)
importances = list(rf_exp.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
[print('Variable(GradientBoostingClassifier) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# rf_exp = RandomForestRegressor(n_estimators=1000, random_state=100)
# rf_exp.fit(X_train, Y_train)
# importances = list(rf_exp.feature_importances_)
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(dataset), importances)]
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# [print('Variable(Random forrest regressor) : {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

model = SelectFromModel(rf_exp, prefit=True)
X_new = model.transform(X_train)
print("Select from model shape: {}".format(X_new.shape))

model = SelectKBest(chi2, k=5)
fit = model.fit(X_train, Y_train)
print("Select k best priorities: {}".format(fit.scores_))
X_new=fit.transform(X_train)
print("Select k best with chi2: {}".format(X_new.shape))

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False,max_iter=2000).fit(X, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print("Linear SVC: {}".format(X_new.shape))
print(model.get_support())

estimator = LogisticRegression(solver='lbfgs')
selector = RFE(estimator, 5)
selector = selector.fit(X, Y)
print("RFE count features: {}, features: {}, ranking: {}".format(selector.n_features_, selector.support_, selector.ranking_))

pca = decomposition.PCA(n_components=12)
fit = pca.fit(X_train, Y_train)
X_new = pca.transform(X_train)
print("PCA: {}".format(X_new.shape))
# print("PCA components: {}".format(fit.components_))
# print("PCA variance: {}".format(fit.explained_variance_ratio_))

ica=FastICA(n_components=12, random_state=0)
x_reduced=ica.fit_transform(X_train)
print("ICA: {}".format(x_reduced.shape))

models = []
# from sklearn.preprocessing import MinMaxScaler
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_validation = scaling.transform(X_validation)

models.append(('LR', LogisticRegression(solver='lbfgs', class_weight="balanced")))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecTree', DecisionTreeClassifier()))  # gini, best
models.append(('NB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('RForest', RandomForestClassifier(n_estimators=100)))
models.append(('Kmeans', KMeans()))
models.append(('Ada', AdaBoostClassifier()))
# models.append(('SVM(linear))', LinearSVC(max_iter=2000)))
# models.append(('SVM', SVC(gamma="scale")))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('SQD', SGDClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('CatBoost', CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')))

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
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# fig = plt.figure(figsize=(10.0, 8.0))
# ax = fig.add_subplot(111)
# plt.boxplot(results, 0, '')
# ax.set_xticklabels(names)
# plt.ylabel("Přesnost")
# for tick in ax.get_xticklabels():
#     tick.set_rotation(70)
# plt.savefig("test_data.png")
with open("test/result_first_test2.txt", "w") as file:
    for name, result in zip(names, results):
        file.write("{},{}\n".format(name, result))
# plt.show()
