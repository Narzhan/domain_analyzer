import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

dataset = pandas.read_csv("result.csv")
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby("label").size())
# dataset.plot(kind='box', subplots=True, layout=(5,6), sharex=False, sharey=False)
# plt.show()
# dataset.hist()
# plt.show()
# scatter_matrix(dataset)
# plt.show()
# Split-out validation dataset


array = dataset.values
X = array[:, 0:12]
Y = array[:, 12]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression(multi_class="auto", solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('RForest', RandomForestClassifier(n_estimators=100)))
models.append(('Kmeans', KMeans()))
models.append(('Ada', AdaBoostClassifier()))
# models.append(('SVM', SGDClassifier()))
models.append(('SVM', SVC(gamma="scale")))

results = []
names = []
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


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()