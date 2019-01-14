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

test_dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
test_dataset = test_dataset.replace(np.nan, '', regex=True)
test_dataset = test_dataset.sort_index()
tfidf = TfidfVectorizer(min_df=0.2, analyzer='word', stop_words="english", ngram_range=(1, 2))
features = tfidf.fit_transform(test_dataset.text)
pickle.dump(features, open("features_array_splitted.pkl", "wb"))


# k means determine k
distortions = []
K = range(5,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(features)
    kmeanModel.fit(features)
    distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Hledání optimální hondoty počtu shluků k')
plt.show()
