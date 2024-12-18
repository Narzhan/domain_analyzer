import pandas
import time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


class MlTester:
    def __init__(self, columns: list):
        self.columns = columns
        self.scoring = 'accuracy'
        self.seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation, self.columes = self.load_data()
        self.result = {}
        self.test_time=""

    def data_properties(self, filename: str):
        dataset = pandas.read_csv(filename, index_col=13)
        print(dataset.shape)
        print(round(dataset.describe(), 2))
        print(dataset.groupby("label").size())
        print("Variances: {}".format(dataset.var()))
        print("Correlations: {}".format(dataset.corr()))

    def prepare_data(self, filename: str):
        return pandas.read_csv(filename, index_col=0, usecols=self.columns)

    def load_data(self):
        dataset = self.prepare_data("dataframe_enhanced.csv")
        array = dataset.values
        X = array[:, 0:-1]
        Y = array[:, -1]
        validation_size = 0.20
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=self.seed, stratify=Y)
        return X_train, X_validation, Y_train, Y_validation, list(dataset)

    def feature_importance_algo(self):
        # algorithms = [ExtraTreesClassifier(n_estimators=50), DecisionTreeClassifier(), GradientBoostingClassifier()]
        algorithms= [RandomForestRegressor(n_estimators=1000, random_state=100)]
        for algorithm in algorithms:
            rf_exp = algorithm
            rf_exp = rf_exp.fit(self.X_train, self.Y_train)
            importances = list(rf_exp.feature_importances_)
            feature_importances = [(feature, round(importance, 2)) for feature, importance in
                                   zip(self.columes, importances)]
            feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            [print('Variable(Extra treee classifier) : {:20} Importance: {}'.format(*pair)) for pair in
             feature_importances]

    def feature_k_best(self, value: int = 5):
        model = SelectKBest(chi2, k=value)
        fit = model.fit(self.X_train, self.Y_train)
        print("Select {} best priorities: {}".format(value, fit.scores_))
        X_new = fit.transform(self.X_train)
        print("Select {} best with chi2: {}".format(value, X_new.shape))

    def feature_linear(self):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000).fit(self.X_train, self.Y_train)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(self.X_train)
        print("Linear SVC: {}".format(X_new.shape))
        print(model.get_support())

    def feature_rfe(self):
        estimator = LogisticRegression(solver='lbfgs')
        selector = RFE(estimator, 5)
        selector = selector.fit(self.X_train, self.Y_train)
        print("RFE count features: {}, features: {}, ranking: {}".format(selector.n_features_, selector.support_,
                                                                         selector.ranking_))

    def reduction_pca(self):
        pca = decomposition.PCA(n_components=len(self.columns)-2)
        fit = pca.fit(self.X_train, self.Y_train)
        self.X_train = pca.transform(self.X_train)
        print(self.X_train)
        # print("PCA: {}".format(self.X_train.shape))
        # print("PCA components: {}".format(fit.components_))
        # print("PCA variance: {}".format(fit.explained_variance_ratio_))

    def reduction_ica(self):
        ica = FastICA(n_components=len(self.columns)-2, random_state=0)
        x_reduced = ica.fit_transform(self.X_train)
        print("ICA: {}".format(x_reduced.shape))

    def models(self):
        return {
                # 'LR': LogisticRegression(solver='lbfgs', class_weight="balanced"),
                # 'LDA': LinearDiscriminantAnalysis(),
                'KNN': KNeighborsClassifier(),
                # 'DecTree': DecisionTreeClassifier(),
                # 'NB': GaussianNB(),
                # 'MNB': MultinomialNB(),
                # 'RForest': RandomForestClassifier(n_estimators=100),
                # 'Ada': AdaBoostClassifier(),
                # 'SVM': LinearSVC(max_iter=2000),
                # 'GBC': GradientBoostingClassifier(),
                # 'SQD': SGDClassifier(max_iter=1000, tol=1e-3),
                # 'XGB': XGBClassifier(),
                # 'CatBoost': CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss',
                #                                verbose=False),
                # 'BNB': BernoulliNB(),
                # 'RC': RidgeClassifier(),
                # 'perc': Perceptron(max_iter=1000, tol=1e-3),
                # 'passive': PassiveAggressiveClassifier(max_iter=1000, tol=1e-3),
                # 'nearest': NearestCentroid(),
                # "LightGGM": lgb.LGBMClassifier(objective="binary", verbose=0)
                }

    def train_best_model(self):
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.Y_train)
        predictions = knn.predict(self.X_validation)
        print(accuracy_score(self.Y_validation, predictions))
        print(confusion_matrix(self.Y_validation, predictions))
        print(classification_report(self.Y_validation, predictions))

    def scale_data_minmax(self):
        scaling = MinMaxScaler(feature_range=(0, 1)).fit(self.X_train)
        self.X_train = scaling.transform(self.X_train)
        self.X_validation = scaling.transform(self.X_validation)

    def scale_data_standartize(self):
        # zero mean, unit variance
        scaler = StandardScaler().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)

    def scale_data_robust(self):
        scaler = RobustScaler().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)


    def scale_data_quantile(self):
        scaler = QuantileTransformer().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)

    def scale_data_power(self):
        scaler = PowerTransformer().fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)

    def scale_data_normalization(self):
        normalizer = Normalizer().fit(self.X_train)
        self.X_train = normalizer.transform(self.X_train)
        self.X_validation = normalizer.transform(self.X_validation)

    def train(self):
        start_time = time.time()
        self.scale_data_minmax()
        for name, model in self.models().items():
            try:
                kfold = model_selection.KFold(n_splits=5, random_state=self.seed)
                cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold,
                                                             scoring=self.scoring)
                print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))
                self.result[name] = cv_results
            except Exception as e:
                print(e)
        print("Took: {}".format(time.time() - start_time))
        self.test_time = time.time() - start_time

    def persist_results(self):
        with open("text/result_first_test{}.txt".format(len(self.columes)), "w") as file:
            file.write("{}\n".format(self.columes))
            file.write("{}\n".format(self.test_time))
            for name, result in self.result.items():
                file.write("{},{}\n".format(name, result))


if __name__ == '__main__':
    columns = ['part_path', 'deep_links', 'fresh', 'pages', 'totalEstimatedMatches', "topics", "tf_idf", "embedding",
               'label', "domain"]
    print(columns)
    tester = MlTester(columns)
    tester.train()
    for column in ['part_path', 'deep_links', 'fresh', 'pages', 'totalEstimatedMatches', "topics", "tf_idf"]:
        columns.remove(column)
        print(columns)
        tester = MlTester(columns)
        tester.train()
    # tester.train_best_model()
    # tester.persist_results()
