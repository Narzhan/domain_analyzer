import os, json

from sklearn.feature_extraction.text import TfidfVectorizer
from tld import get_tld

# result = {}
# top_level = ["_type", "queryContext", "webPages", "rankingResponse", "relatedSearches"]
# web_level = ["id", "name", "url", "about", 'dateLastCrawled', 'snippet', 'isNavigational', 'language',
#              'isFamilyFriendly', 'displayUrl']
# for path, label in {"D:/Narzhan/Documents/dipl/data/clean/data/": 0,
#                     "D:/Narzhan/Documents/dipl/data/malicious/data/": 1}.items():
#     for file in os.listdir(path):
#         try:
#             with open("{}{}".format(path, file), "r") as base:
#                 domain = file.replace(".json", "")
#                 domain_tld = get_tld(domain, as_object=True, fix_protocol=True)
#                 data = json.loads(base.read())
#                 result[domain] = {}
#                 for key in data:
#                     if key not in top_level:
#                         try:
#                             result[domain]["top_level"].append(key)
#                         except KeyError:
#                             result[domain]["top_level"] = [key]
#                 if "webPages" in data and len(data["webPages"]["value"]) > 0:
#                     for page in data["webPages"]["value"]:
#                         match = "uknown"
#                         try:
#                             url_tld = get_tld(page["url"], as_object=True, fix_protocol=True)
#                         except Exception:
#                             pass
#                         else:
#                             if domain_tld.fld == url_tld.fld:
#                                 match = True
#                             else:
#                                 match = False
#                         for k in page:
#                             if k not in web_level:
#                                 try:
#                                     result[domain]["web_level"].append([k, match])
#                                 except KeyError:
#                                     result[domain]["web_level"] = [[k, match]]
#                 if len(result[domain]) == 0:
#                     del result[domain]
#         except Exception as e:
#             print("File error, {}, {}".format(file, e))
# with open("carl_result", "w") as output:
#     json.dump(result, output)


# with open("carl_result.json", "r") as file:
#     data = json.loads(file.read())
#
# for k, v in data.copy().items():
#     if len(v) == 1 and "web_level" in v:
#         for item in v["web_level"]:
#             if item[0] == "deepLinks":
#                 data[k]["web_level"].remove(item)
# top_counter = {}
# for k, v in data.items():
#     if "top_level" in v:
#         for cat in v["top_level"]:
#             try:
#                 top_counter[cat] += 1
#             except KeyError:
#                 top_counter[cat] = 1
#
# web_counter = {"true": {}, "false": {}}
# for k, v in data.items():
#     if "web_level" in v:
#         for cat in v["web_level"]:
#             logical = None
#             if cat[1] is True:
#                 logical = "true"
#             else:
#                 logical = "false"
#             try:
#                 web_counter[logical][cat[0]] += 1
#             except KeyError:
#                 web_counter[logical][cat[0]] = 1
# print(top_counter)
# print(web_counter)



from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas, pickle

# test_dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
# test_dataset = test_dataset.replace(np.nan, '', regex=True)
# test_dataset = test_dataset.sort_index()
# tfidf = TfidfVectorizer(min_df=0.2, analyzer='word', stop_words="english", ngram_range=(1, 2))
# features = tfidf.fit_transform(test_dataset.text)
# pickle.dump(features, open("features_array_splitted.pkl", "wb"))
#
#
# # k means determine k
# distortions = []
# K = range(5,20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(features)
#     kmeanModel.fit(features)
#     distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])
#
# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('Hledání optimální hondoty počtu shluků k')
# # plt.show()
# from numpy import asarray
# embeddings_index = {}
# with open('pretrained/wiki.en.vec', "r", encoding="utf-8") as file:
#     next(file)
#     for line in file:
#         values = line.split()
#         # if len(values) > 301:
#         #     values.pop(1)
#         #     print("popped")
#         word = values[0]
#         coefs = asarray(values[len(values)-300:], dtype='float32')
#         embeddings_index[word] = coefs
# # f = open('pretrained/wiki.cs.vec', encoding="utf-8")
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# incorrect_domins = ["usa.cc", "turkhackteam.org", "nut.cc", "ferozo.com", "criticbay.com", "servicemarket.su",
#                     "quanmama.com", "byethost13.com", "beget.tech", "forumattivo.it", "newtonpaiva.br", "cpvdo.com",
#                     "hpsd.k12.pa.us", "eldivisadero.cl", "seriesgato.tv", "roamans.com", "paisabazaar.com",
#                     "okhatrimaza.org", "paypal-customerfeedback.com", "saldao-mes-das-criancas.com", "lomo.jp",
#                     "dresk.ru", "umbler.net", "userproplugin.com", "seriesgato.com", "tvoe-zoloto.com", "unaux.com",
#                     "perfectliker.com", "statichtmlapp.com", "wefbee.com", "crkphotoimaging.com.au", "hatenablog.com",
#                     "aasaanjobs.com", "copymethat.com", "jigsy.com", "celitel2.com", "4gram.com",
#                     "epochtimeschicago.com", "xsph.ru", "creditonebank.com", "tc-clicks.com", "jeun.fr",
#                     "byethost8.com", "flazio.com", "tripod.com", "doctr1ne.com", "ulcraft.com", "live.com",
#                     "likesgroup.com", "filmshared.com", "19tv.top", "naijaextra.com", "capitalcu.com", "cewomen.com",
#                     "pfashionmart.com", "webhostbox.net", "7m.pl", "begambleaware.org", "dunya.com.pk",
#                     "exploreourapps.com", "uol.com.br", "surveygizmo.com", "fbsub.de", "byethost32.com", "zz.am",
#                     "nacosti.go.ke", "ultimatefreehost.in", "idgod.ph", "dotapicker.com", "cuasotinhyeu.vn",
#                     "yamadadenkishop.com", "adexten.com", "byethost12.com", "igg.biz", "zzz.com.ua",
#                     "playstationmail.net", "linkbax.com", "watchcric.eu", "traktrafficflow.com", "tumblr.com",
#                     "my-free.website", "clickconfirmation.com", "2090000.ru", "e-monsite.com", "telegra.ph",
#                     "emailaccessonline.com", "vipfb.es", "officialliker.co", "comuesp.com", "byethost7.com",
#                     "hostland.pro", "webxion.com", "myartsonline.com", "toexten.com", "icloudbaypass.com", "ukit.me",
#                     "2go.com.ph", "cabanova.com", "strikingly.com", "yola.com", "pe.hu", "cla.fr", "somee.com", "es.tl",
#                     "dreamscape317.net", "html-5.me"]
# remove = {'barnygeeft.byethost6.com', 'byethost9.com', 'byethost.com'}
from keras.utils import plot_model
from keras.models import load_model
from IPython.display import SVG

import conx as cx
model = load_model("pca/standard_model_nn.h5")
SVG(model_to_dot(model).create(prog='dot', format='svg'))




# dataset_lda = pandas.read_csv("dataframe_enhanced.csv", index_col=0)
# for domain in incorrect_domins:
#     line= dataset_lda.loc[domain]
#     if line["label"] == 1.0:
#         try:
#             os.remove("D:/Narzhan/Documents/dipl/data/clean/data/{}.json".format(domain))
#         except Exception as e:
#             print(e)
#     else:
#         try:
#             os.remove("D:/Narzhan/Documents/dipl/data/malicious/data/{}.json".format(domain))
#         except Exception as e:
#             print(e)

# dataset_lda = pandas.read_csv("splitted_text/lda/result_data.csv", index_col=0)
# dataset_tfidf = pandas.read_csv("splitted_text/tf_idf/result_data.csv", index_col=0)


# domain_mapping = {}
# numeric_dataset = pandas.read_csv("test_data.csv", index_col=12)
# for domain in incorrect_domins:
#     domain_mapping[domain] = numeric_dataset.loc[domain]["label"]
# import csv
# temp = set()
# with open("text_test_data_splitted.csv", "r", encoding="utf-8") as file:
#     with open("text_test_data_splitted_fixed.csv", "w", encoding="utf-8") as outfile:
#         reader = csv.reader(file, delimiter=";")
#         # outfile.write("{}\n".format(";".join(next(reader))))
#         for row in reader:
#             if row[3] in domain_mapping and int(row[2]) != domain_mapping[row[3]]:
#                 temp.add(row[3])
#             else:
#                 outfile.write("{}\n".format(";".join(row)))
#
# if len(temp) != len(incorrect_domins):
#     print(temp)
# from keras import Input, Model
# from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, GRU, Bidirectional, GlobalMaxPool1D
#
# t = pickle.load(open("splitted_text/word_embedding/tokenizer.pkl", "rb"))
# vocab_size = len(t.word_index) + 1
# t=None
# MAX_LEN = 134
# input_layer = Input((MAX_LEN,))
#
# # Add the word embedding Layer
# embedding_layer = Embedding(vocab_size, 100)(input_layer)
# embedding_layer = SpatialDropout1D(0.4)(embedding_layer)
#
# # Add the LSTM Layer
# lstm_layer = LSTM(25)(embedding_layer)
# pooling = GlobalMaxPool1D()(embedding_layer)
# # Add the output Layers
# output_layer1 = Dense(50, activation="relu")(lstm_layer)
# output_layer1 = Dropout(0.4)(output_layer1)
# output_layer2 = Dense(1, activation="sigmoid")(output_layer1)
#
# # Compile the model
# model = Model(inputs=input_layer, outputs=output_layer2)
# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
# model.save("no_embedding.h5")