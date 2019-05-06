import pandas as pd
import numpy as np
# domains = []
# with open("test_data.txt", "r") as file:
#     reader = csv.reader(file)
#     for row in reader:
#         domains.append(row[0])

# first = pd.read_csv("features.csv", index_col=0)
# second = pd.read_csv("part2/features.csv", index_col=0)
# third = pd.read_csv("part3/features.csv", index_col=0)
# # fourth = pd.read_csv("part3/results2.csv", index_col=0)
# result = pd.concat([first, second, third])
# result2 = result.loc[result.index.isin(domains)]
# print(result2)
# result2.to_csv("features_final.csv")

# results = pd.read_csv("test_combined.csv", index_col=0)
# del results["label"]
# new = pd.read_csv("results.csv", index_col=0)
# print(len(new))
# result = pd.concat([results, new])
# labels = pd.read_csv("test_data.txt", index_col=0)
# dataset = result.join(labels)
# print(len(dataset))
# dataset.to_csv("test_combined.csv")
# features = pd.read_csv("features_final.csv", index_col=0)
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

dataset = pd.read_csv("test_combined.csv", index_col=0)
# for method in ["knn", "lsvc", "rforest", "lightgbm", "nn"]:
#     print(method)
#     print(accuracy_score(dataset["label"], dataset[method]))
#     print(confusion_matrix(dataset["label"], dataset[method]))
#     print(classification_report(dataset["label"], dataset[method]))
#
# counter = 0
# correct = {}
# incorrect = {}
# for input, prediction, label in zip(dataset.index, dataset["nn"], dataset["label"]):
#     try:
#         row = features.loc[[input]]
#         if prediction != label:
#             incorrect[input] = {"tf_idf": int(row["tf_idf"]), "lda": int(row["topics"]), "we": int(row["embedding"]), "label": label}
#         else:
#             correct[input] = {"tf_idf": int(row["tf_idf"]), "lda": int(row["topics"]), "we": int(row["embedding"]), "label": label}
#     except Exception:
#         pass
# #     counter += 1
# import json
# with open("correct.json", "w") as file:
#     json.dump(correct, file)
# with open("incorrect.json", "w") as file:
#     json.dump(incorrect, file)
# for method in ["knn", "rforest", "lightgbm", "nn"]:
#     correct = {}
#     incorrect = {}
#     for input, prediction, label, prob in zip(dataset.index, dataset[method], dataset["label"], dataset["{}_prob".format(method)]):
#         if prediction != label:
#             incorrect[input] = prob
#             # print(prediction, prob, label)
#         else:
#             correct[input] = prob
#     import matplotlib.pyplot as plt
#     # hist
#     plt.hist(list(correct.values()), bins=20, color='blue')
#     plt.xticks(np.arange(0, 1, 0.05), rotation=70)
#     plt.xlabel('Probability')
#     # plt.figure(figsize=(10,8))
#     plt.savefig("{}.png".format(method))
#     plt.gcf().clear()
# import json
#
# features = {"tf_idf": {"correct": 0, "incorrect": 0}, "lda": {"correct": 0, "incorrect": 0},
#             "we": {"correct": 0, "incorrect": 0}}
# # with open("correct.json", "r") as file:
# #     data = json.load(file)
# #     for domain, status in data.items():
# #         for section in features.keys():
# #             if status[section] == status["label"]:
# #                 features[section]["correct"] += 1
# #             else:
# #                 features[section]["incorrect"] += 1
# with open("incorrect.json", "r") as file:
#     data = json.load(file)
#     for domain, status in data.items():
#         for section in features.keys():
#             if status[section] == status["label"]:
#                 features[section]["correct"] += 1
#             else:
#                 features[section]["incorrect"] += 1
# print(features)
# import matplotlib.pyplot as plt
# features={'tf_idf': {'incorrect': 4876, 'correct': 45115}, 'we': {'incorrect': 10318, 'correct': 39673}, 'lda': {'incorrect': 5393, 'correct': 44598}}
# correct = [features["tf_idf"]["correct"], features["lda"]["correct"], features["we"]["correct"]]
# incorrect = [features["tf_idf"]["incorrect"], features["lda"]["incorrect"], features["we"]["incorrect"]]
# # fig = plt.figure()
# # ax1 = fig.add_subplot(121)
# # ax1.plot(correct)
# # ax2 = fig.add_subplot(122)
# # ax2.plot(incorrect)
# # plt.show()
# # x1 = list(dataset[dataset['label'] == 1][cat])
# # x2 = list(dataset[dataset['label'] == 0][cat])
# names = ["tf_idf", "we", "lda"]
# plt.bar(names, correct, color = 'b')
# plt.bar(names, incorrect, color = 'r', bottom = correct)
# plt.show()
#
# from tld import get_tld
# import re
# import csv
# pattern = re.compile("(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]")
# with open("test_data.txt", "r") as file:
#     reader = csv.reader(file)
#     for row in reader:
#         try:
#             if not pattern.match(row[0]):
#                 print(row)
#             # domain_tld = get_tld(row[0], as_object=True, fix_protocol=True)
#             # test =domain_tld.fld
#         except Exception as e:
#             print(e)
#             print(row)
# import json
# with open("results.json", "r") as file:
#     data = json.load(file)
# correct, incorrect= 0,0
# for key, value in data.items():
#     label = results.loc[[key]]
#     if int(label["label"]) == value:
#         correct+=1
#     else:
#         incorrect+=1
# print(correct)
# print(incorrect)
# import json
# import numpy as np
# with open("results.json", "r") as file:
#     data = json.load(file)
# predicted = pd.DataFrame(list(data.values()), columns=["test"], index=list(data.keys()))
# print(predicted)
# result = results.join(predicted)
# result = result.replace(np.nan, 0, regex=True)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# print(result)
# print(accuracy_score(result["label"], result["test"]))
# print(confusion_matrix(result["label"], result["test"]))
# print(classification_report(result["label"], result["test"]))
# for mwthod in ["knn", "rforest", "lightgbm", "nn"]:
#     print(mwthod)
#     print("###################################")
#     for number in np.arange(0.5, 1, 0.1):
#         print(number)
#         new_labels = []
#         for prediction, label, prob in zip(dataset[mwthod], dataset["label"], dataset["{}_prob".format(mwthod)]):
#             if prob < number:
#                 new_labels.append(0)
#             else:
#                 new_labels.append(prediction)
#         print(accuracy_score(dataset["label"], new_labels))
#         print(confusion_matrix(dataset["label"], new_labels))
#         print(classification_report(dataset["label"], new_labels))
import os, pickle
#
features = pd.read_csv("features_final.csv", index_col=0)
del features["embedding"]
we_features = pd.read_csv("we_prob_predictions.csv", index_col=0)
features["embeddings"] = we_features["pred"]
final = features.join(dataset["label"])
# print(final)
# print(final)
# print(accuracy_score(final["label"], final["tf_idf"]))
# print(confusion_matrix(final["label"], final["tf_idf"]))
# print(classification_report(final["label"], final["tf_idf"]))
# print(final)
from keras.models import load_model
array = final.values
X = array[:, 0:-1]
Y = array[:, -1]
# for name in ["standard"]:
scaler = pickle.load(open("pca/standard.pkl", "rb"))
scaled = scaler.transform(X)
pca = pickle.load(open("pca/pca.pkl", "rb"))
scaled = pca.transform(scaled)
# print(name)
# print("###############")
for model_name in ["lightgbm.pkl", "knn", "forests", "lsvc"]:
    model = pickle.load(open("pca/standard_model_{}".format(model_name), "rb"))
    predict = model.predict(scaled)
    print(model_name)
    print(accuracy_score(Y, predict))
    print(confusion_matrix(Y, predict))
    print(classification_report(Y, predict))
model = load_model("pca/standard_model_nn.h5")
predict = model.predict(scaled)
predicted = (predict > 0.5)
print("nn")
print(accuracy_score(Y, predicted))
print(confusion_matrix(Y, predicted))
print(classification_report(Y, predicted))
# scaler = pickle.load(open("scaler.pkl", "rb"))
# X = scaler.transform(X)
#
# for name in os.listdir("binaries/"):
#     print(name)
#     clssifier = pickle.load(open("binaries/{}".format(name), "rb"))
#     predicted = clssifier.predict(X)
#     try:
#         predict_prob = [max(estimate) for estimate in clssifier.predict_proba(X)]
#     except Exception:
#         pass
#     else:
#         counter = 0
#         for prob in predict_prob:
#             if prob < 0.95:
#                 predicted[counter] = 0
#             counter+=1
#         print(accuracy_score(Y, predicted))
#         print(confusion_matrix(Y, predicted))
#         print(classification_report(Y, predicted))
# nn = load_model("dense_model_diff.h5")
# predicted = nn.predict(X)
# predicted = (predicted > 0.5)
# print(accuracy_score(Y, predicted))
# print(confusion_matrix(Y, predicted))
# print(classification_report(Y, predicted))

# import matplotlib.pyplot as plt
# label_ids = dataset["label"]
# predictions = dataset["nn"]
# tsne = TSNE(n_components=2, perplexity=40.0)
# data_2d = tsne.fit_transform(features.values)
# np.save("2d_data", data_2d)
# plt.figure(figsize=(20,20))
# plt.grid()
#
# nb_classes = len(np.unique(label_ids))
#
# for label_id in np.unique(label_ids):
#     plt.scatter(data_2d[np.where(label_ids == predictions), 0],
#                 data_2d[np.where(label_ids == predictions), 1],
#                 marker='o',
#                 color=plt.cm.Set1(label_id / float(nb_classes)),
#                 linewidth='1',
#                 alpha=0.8,
#                 label=label_id)
#     plt.scatter(data_2d[np.where(label_ids == label_id), 0],
#                 data_2d[np.where(label_ids == label_id), 1],
#                 marker="v",
#                 color=plt.cm.Set1(label_id / float(nb_classes)),
#                 linewidth='1',
#                 alpha=0.8,
#                 label=label_id)
# plt.legend(loc='best')
# plt.show()
# new = pd.read_json("padded_texts.json", orient="index")
# new = new.join(dataset["label"])
# print(round(new.describe(), 2))
# new = new.loc[new['length'] == 10]
# print(new.groupby("label").size())
# print(new)