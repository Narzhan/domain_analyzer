import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report


def load_data(file: str):
    dataset = pd.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
    dataset = dataset.replace(np.nan, '', regex=True)
    dataset = dataset.sort_index()
    source_dataset = pd.read_csv(file, index_col=0, encoding='utf-8')
    source_dataset = source_dataset.sort_index()
    # feature_array = pickle.load(open(file, "rb"))
    # feature_array = np.load(file)
    mapper = {}
    domain_mapping = {}
    row_id = 0
    for i, row in dataset.iterrows():
        try:
            mapper[i].append(row_id)
        except KeyError:
            mapper[i] = [row_id]
        # domain_mapping[i] = {}
        domain_mapping[i] = {"label": row["label"]}
        row_id += 1
    dataset, row_id = None, None
    print("labels done")
    for domain, id_list in mapper.items():
        temp_list = []
        for ids in id_list:
            temp_list.append(source_dataset.iloc[ids]["we"])
            # temp_list.extend(feature_array[ids].toarray()[0])
            # temp_list.extend(feature_array[ids])
        domain_mapping[domain]["features"] = tuple(map(lambda x: 1 if x else 0, temp_list))
        # domain_mapping[domain]["features"] = tuple(temp_list)
    print("features done")
    feature_array, mapper = None, None
    # pickle.dump(domain_mapping, open("domain_mapping.pkl", "wb"))
    # final_dataset = pd.DataFrame(list(domain_mapping.values()), index=domain_mapping.keys())
    # tmp = final_dataset.features.apply(pd.Series)
    # final_dataset.drop(["features"], axis=1)
    # final_dataset = tmp.merge(final_dataset, right_index=True, left_index=True)
    # final_dataset = final_dataset.features.apply(pd.Series).merge(final_dataset, right_index=True, left_index=True).drop(["features"], axis=1)
    temp_dataset = pd.DataFrame(list(domain_mapping.values()), index=list(domain_mapping.keys()))
    final_dataset = pd.DataFrame(temp_dataset['features'].values.tolist(), index=temp_dataset.index.tolist())
    final_dataset["label"] = temp_dataset["label"]
    final_dataset = final_dataset.replace(np.nan, 0, regex=True)
    print("final done")
    return final_dataset


def models():
    print("Test started")
    models = []
    models.append(('BMeta', BaggingClassifier()))
    models.append(('RForest', RandomForestClassifier(n_estimators=100)))
    models.append(('Ada', AdaBoostClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('XGB', XGBClassifier()))
    models.append(('CatBoost', CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')))
    scoring = "accuracy"
    results = []
    names = []
    for name, model in models:
        try:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, n_jobs=3)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        except Exception as e:
            print(e)
        else:
            print(msg)


def train_model(name: str):
    classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifier.fit(X_train, Y_train)
    X_predicted = classifier.predict(X_train)
    predictions = classifier.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    result_dataset = pd.DataFrame(data=np.concatenate((X_predicted, predictions), axis=0),
                                  index=np.concatenate((domains_train, domains_test), axis=0), columns=[name])
    result_dataset.to_csv("splitted_text/tf_idf/result_data.csv")
    pickle.dump(classifier, open("splitted_text/tf_idf/model.pkl", "wb"))


def hyper_parameters():
    params = {
        "n_estimators": [x * 100 for x in range(1, 16)],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 8, 12, 16, 32],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "min_impurity_decrease": [0, 0.1, 0.2, 0.3, 0.5],
        "oob_score": [False, True],
        "class_weight": ["balanced", None, "balanced_subsample"]
    }
    classifier = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=classifier, param_distributions=params, n_iter=500, cv=3, verbose=2,
                                   random_state=7, n_jobs=-1, scoring="accuracy")
    rf_random.fit(X_train, Y_train)
    print(rf_random.best_params_)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random)


def hyper_parameters_exh():
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    classifier = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring="accuracy",
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid)


def evaluate(model):
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def evaluate_model(name: str):
    classifier = pickle.load(open(name, "rb"))
    predictions = classifier.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def light_gbm():
    # train_data = lgb.Dataset(X_train, label=Y_train)
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_validation, Y_validation, reference=lgb_train)
    # param = {'num_leaves': 31, 'num_trees': 50, 'objective': 'binary', 'metric': 'auc'}
    param = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'auc', "binary_logloss"},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    bst = lgb.train(param, lgb_train, 20, valid_sets=lgb_eval)
    y_pred = bst.predict(X_validation, num_iteration=bst.best_iteration)
    for i in range(0, Y_validation.shape[0]):
        if y_pred[i] >= .5:  # setting threshold to .5
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_validation, y_pred)
    print(cm)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_pred, Y_validation)
    print("LightGBM: {}".format(accuracy))
    # print('The mean of prediction is:', y_pred.mean(), y_pred.std())


for model in ["bilstm_fasttext_mixed", "lstm_w2v_custom", "lstm_w2v_mixed", "no_embedding_trained_bias_l1"]:
    print("Going over {} model".format(model))
    print("#######################################")
    dataset = load_data("preprocessed/{}.csv".format(model))
    dataset.to_csv("preprocessed/dataframe_{}.csv".format(model))
    # dataset = pd.read_csv("{}.csv".format(model), index_col=0, encoding='utf-8')
    array = dataset.values
    domains = dataset.index.values
    dataset = None
    X = array[:, 0:-1]
    Y = array[:, -1]
    print("array created")
    array = None
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation, domains_train, domains_test = model_selection.train_test_split(X, Y,
                                                                                                                 domains,
                                                                                                                 test_size=validation_size,
                                                                                                                 random_state=seed)
    X, Y = None, None
    # evaluate_model("splitted_text/lda/model.pkl")
    light_gbm()
    models()
