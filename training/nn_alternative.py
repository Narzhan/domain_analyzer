import json
import pickle
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
# ensamble_we = pickle.load(open("ensamble.pkl", "rb"))
# model = load_model("we_model.h5")
test_dataset = pd.read_csv("test_combined.csv", index_col=0)

# results = {}
# raw_predictions = {}
# counters = {"correct": 0, "incorrect": 0}
results = {}
# with open("lstm_predictions.csv", "w") as outfile:
#     with open("lstm_predictions_prob.csv", "w") as outfile_prob:
for domain in test_dataset.index:
    try:
        with open("D:/Narzhan/Documents/dipl/data/test_data/{}.json".format(domain), "r", encoding="utf-8") as file:
            texts = []
            if os.stat("D:/Narzhan/Documents/dipl/data/test_data/{}.json".format(domain)).st_size != 0:
                data = json.load(file)
                if "webPages" in data:
                    for page in data["webPages"]["value"]:
                        texts.append(page["snippet"])
            while len(texts) < 10:
                texts.append("")
            results[domain] = texts
            # padded_texts = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=134)
            # results[domain] = padded_texts
            # all_zero = np.where(~padded_texts.any(axis=1))[0]
            # results[domain] = {"positions": all_zero.tolist(), "length": len(all_zero)}
            # predictions = model.predict(padded_texts)
            # raw_predictions[domain] = predictions
            # features = [list(map(lambda x: 1 if x > 0.5 else 0, predictions))]
            # outfile.write("{},{}\n".format(domain, ",".join(str(x) for x in features)))
            # outfile_prob.write("{},{}\n".format(domain, ",".join(str(x) for x in predictions)))
            # predictions = ensamble_we.predict(features)[0]
            # results[domain] = predictions
            # label = dataset.loc[[domain]]
            # if int(label["label"]) == predictions:
            #     counters["correct"] += 1
            # else:
            #     counters["incorrect"] += 1
    except Exception as e:
        print(e)

train_dataset = pd.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";",
                            engine="python")
train_dataset = train_dataset.replace(np.nan, '', regex=True)
train_dataset = train_dataset.sort_index()

for name, model_path in {"pretrained": "updated_fasttext/bilstm_embedding_martix.h5",
                         "custom": "updated_fasttext/bilstm_embedding_martix_custom.h5",
                         "aligned": "updated_fasttext/bilstm_embedding_martix_custom_al.h5",
                         "nearmiss": "weights/no_bi_embedding_trained_reg_nearmiss.h5",
                         "smote": "weights/no_bi_embedding_trained_reg_smote.h5",
                         "weighted": "weights/no_bi_embedding_trained_reg_weighted.h5",
                         }.items():
    model = load_model(model_path)
    train = {}
    for index, row in train_dataset.iterrows():
        try:
            train[index]["data"].append(row["text"])
        except KeyError:
            train[index] = {"data": [row["test"]], "label": row["label"]}

    X, Y = [], []
    for domain, data in train.items():
        padded_texts = pad_sequences(tokenizer.texts_to_sequences(data["data"]), maxlen=134)
        predictions = model.predict(padded_texts)
        X.append(list(map(lambda x: 1 if x > 0.5 else 0, predictions)))
        Y.append(data["label"])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7,
                                                                                    stratify=Y)
    print(name)
    rand_f = RandomForestClassifier(n_estimators=100)
    rand_f.fit(X_train, Y_train)
    pickle.dump(rand_f, open("ensemble/rforest_{}.pkl".format(name), "wb"))
    predition = rand_f.predict(X_validation)
    print(accuracy_score(Y_validation, predition))
    print(confusion_matrix(Y_validation, predition))
    print(classification_report(Y_validation, predition))

    print("----------")
    X_test, Y_test = [], []
    for index, row in test_dataset.iterrows():
        try:
            predictions = model.predict(pad_sequences(tokenizer.texts_to_sequences(results[index]), maxlen=134))
            X_test.append(list(map(lambda x: 1 if x > 0.5 else 0, predictions)))
            Y_test.append(data["label"])
        except Exception as e:
            print(e)
    predition = rand_f.predict(X_test)
    print(accuracy_score(Y_test, predition))
    print(confusion_matrix(Y_test, predition))
    print(classification_report(Y_test, predition))


# with open("test_texts.json", "w", encoding="utf-8") as file:
#     json.dump(results, file, indent=4)