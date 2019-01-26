import pandas, nltk
import numpy as np
from gensim.models import word2vec
from gensim.models import FastText
import gensim, pickle
from keras.preprocessing.text import Tokenizer
from nltk import SnowballStemmer, WordNetLemmatizer

# dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
# dataset = dataset.replace(np.nan, '', regex=True)
# dataset = dataset.sort_index()
# stemmer = SnowballStemmer('english')
#
# def lemmatize_stemming(text):
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
#
#
# def preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             result.append(lemmatize_stemming(token))
#     return result
#
# # processed_docs = pickle.load(open("gensim_lda/processed_docs.pkl", "rb"))
# processed_docs = dataset['text'].map(preprocess)
#
#
# feature_size = 100  # Word vector dimensionality
# window_context = 20  # Context window size
# min_word_count = 10  # Minimum word count
# sample = 1e-3  # Downsample setting for frequent words
#
# # w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
# #                               window=window_context, min_count=min_word_count,
# #                               sample=sample, iter=10)
# fast_text = FastText(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=50)
# fast_text.save("fast_text.pkl")
# # w2v_model = word2vec.Word2Vec.load("gensim_we/w2v_model.pkl")
#
#
#
# # similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
# #                   for search_term in ['malware', 'phishing', 'hoax', 'exploit', 'botnet', 'spam']}
# # print(similar_words)
#
# def average_word_vectors(words, model, vocabulary, num_features):
#     feature_vector = np.zeros((num_features,), dtype="float64")
#     nwords = 0.
#
#     for word in words:
#         if word in vocabulary:
#             nwords = nwords + 1.
#             feature_vector = np.add(feature_vector, model[word])
#
#     if nwords:
#         feature_vector = np.divide(feature_vector, nwords)
#
#     return feature_vector
#
#
# def averaged_word_vectorizer(corpus, model, num_features):
#     vocabulary = set(model.wv.index2word)
#     features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
#                 for tokenized_sentence in corpus]
#     return np.array(features)
#
#
# w2v_feature_array = averaged_word_vectorizer(corpus=processed_docs, model=fast_text,
#                                              num_features=feature_size)
# # document_array = pandas.DataFrame(w2v_feature_array)
# pickle.dump(w2v_feature_array, open("fasttext_feature_array.pkl", "wb"))
# # w2v_feature_array = pickle.load(open("w2v_feature_array.pkl", "rb"))
# # w2v_model = None
# from sklearn.cluster import AffinityPropagation, MiniBatchKMeans
#
# # ap = AffinityPropagation()
# ap = MiniBatchKMeans()
# ap.fit(w2v_feature_array)
# cluster_labels = ap.labels_
# pickle.dump(cluster_labels, open("cluster_labels_fasttext.pkl", "wb"))
# dataset.reset_index(drop=True, inplace=True)
# cluster_labels = pandas.DataFrame(cluster_labels, columns=['ClusterLabel'])
# print(pandas.concat([dataset, cluster_labels], axis=1))

from numpy import asarray

docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# embeddings_index = {}
# f = open('glove.6B.100d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# t = Tokenizer()
# t.fit_on_texts(docs)
# vocab_size = len(t.word_index) + 1
# embedding_matrix = np.zeros((vocab_size, 300))
# print(t.word_index)
# for word, i in t.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

######################################
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 300))
with open("", "r") as file:
    for line in file:
        values = line.split()
        if values[0] in t.word_index:
            word = values[0]
            coefs = asarray(values[len(values)-300:], dtype='float32')
            embedding_matrix[t.word_index[word]] = coefs


# switch process, first load text data than load embedding and filter it accordingly to the present text documents