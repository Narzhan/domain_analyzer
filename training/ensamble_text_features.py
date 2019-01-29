import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import gc

# array = dataset.values
# X = array[:, 0:10]
# Y = array[:, 10]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation, indices_train, indices_test = model_selection.train_test_split(X, Y,
#                                                                                                              dataset.index,
#                                                                                                         test_size=validation_size,
#                                                                                                              random_state=seed)


def load_data(file: str):
    dataset = pd.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";",
                              engine="python")
    dataset = dataset.replace(np.nan, '', regex=True)
    dataset = dataset.sort_index()
    feature_array = pickle.load(open(file, "rb"))
    mapper = {}
    domain_mapping = {}
    row_id = 0
    for i, row in dataset.iterrows():
        try:
            mapper[i].append(row_id)
        except KeyError:
            mapper[i] = [row_id]
        domain_mapping[i] = {}
        #domain_mapping[i] = {"label": row["label"]}
        row_id += 1
    dataset, row_id = None, None
    print("labels done")
    for domain, id_list in mapper.items():
        temp_list = []
        for ids in id_list:
            temp_list.extend(feature_array[ids].toarray()[0])
        domain_mapping[domain]["features"] = tuple(temp_list)
    print("features done")
    feature_array, mapper = None, None
    gc.collect()
    pickle.dump(domain_mapping, open("domain_mapping.pkl", "wb"))
    final_dataset = pd.DataFrame(list(domain_mapping.values()), index=domain_mapping.keys())
    #tmp = final_dataset.features.apply(pd.Series)
    #final_dataset.drop(["features"], axis=1)
    #final_dataset = tmp.merge(final_dataset, right_index=True, left_index=True)
    final_dataset = final_dataset.features.apply(pd.Series).merge(final_dataset, right_index=True, left_index=True).drop(["features"], axis=1)
    #temp_dataset = pd.DataFrame(list(domain_mapping.values()), index=list(domain_mapping.keys()))
    #final_dataset = pd.DataFrame(temp_dataset['features'].values.tolist(), index=list(domain_mapping.keys()))
    #final_dataset["label"] = temp_dataset["label"]
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
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring, n_jobs=-1)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        except Exception as e:
            print(e)
        else:
            print(msg)

for i in range(2,6):
    print("Going over {}% min df".format(i))
    print("#######################################")
    dataset = load_data("tf_idf/features_{}.pkl".format(i))
    dataset.to_csv("tf_idf/dataframe{}.csv".format(i))
    array = dataset.values
    dataset=None
    X = array[:, 0:-1]
    Y = array[:, -1]
    print("array created")
    array = None
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size,
                                                                                                                   random_state=seed)
    X, Y = None, None
    models()
