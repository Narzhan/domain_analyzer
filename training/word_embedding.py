import pandas, nltk
import numpy as np
import string
from gensim.models import word2vec
from gensim.models import FastText
import gensim, pickle
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from nltk import SnowballStemmer, WordNetLemmatizer
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
from sklearn.metrics import confusion_matrix

dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
languages = ["en", "cs", "de", "es", "fr", "ja", "ru", "zh"]
# pretrained_dataset = dataset.loc[dataset["language"].isin(languages)]
customtrain_dataset = dataset.loc[~dataset["language"].isin(languages)]


def preprocess(text):
    # convert to list the input
    result = []
    for line in text:
        tokens = [key.lower() for key in nltk.word_tokenize(line)]
        table = str.maketrans("", "", string.punctuation)
        stripped = [key.translate(table) for key in tokens]
        words = [word for word in stripped if word.isalpha()]
        result.append(words)
    return result


processed_docs = customtrain_dataset['text'].map(preprocess)
feature_size = 300
window_context = 10
min_word_count = 5
sample = 1e-3

fast_text = FastText(processed_docs, size=feature_size,
                     window=window_context, min_count=min_word_count, sample=sample,
                     iter=50)
fast_text.save("splitted_text/word_embedding/fast_text/fast_text.pkl")
fast_text.wv.save_word2vec_format("pretrained/custom_embedding.txt", binary=False)


t = Tokenizer()
t.fit_on_texts(dataset.text)
vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 300))
files_location = ["wiki.cs.vec", "wiki.de.vec", "wiki.en.vec", "wiki.es.vec", "wiki.fr.vec", "wiki.ja.vec",
                  "wiki.ru.vec", "wiki.zh.vec"]
for file_location in files_location:
    with open("pretrained/{}".format(files_location), "r") as file:
        for line in file:
            values = line.split()
            if values[0] in t.word_index:
                word = values[0]
                coefs = asarray(values[len(values)-300:], dtype='float32')
                embedding_matrix[t.word_index[word]] = coefs

with open("pretrained/custom_embedding.txt", "r") as file:
    for line in file:
        values = line.split()
        if values[0] in t.word_index:
            word = values[0]
            coefs = asarray(values[len(values) - 300:], dtype='float32')
            embedding_matrix[t.word_index[word]] = coefs
np.save("splitted_text/word_embedding/fast_text/embedding_martix.npy", embedding_matrix)

X_train, y_train, X_validation, y_validation= None, None, None,None

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = Input((70,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    model.fit(X_train, y_train, batch_size=100, nb_epoch=10)
    evaluate_model(model)


def create_rnn_gru():
    # Add an Input Layer
    input_layer = Input((70,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    model.fit(X_train, y_train, batch_size=100, epochs=10)
    evaluate_model(model)


def create_bidirectional_rnn(X_train, y_train):
    # Add an Input Layer
    input_layer = Input((70,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = Bidirectional(GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    model.fit(X_train, y_train, batch_size=100, nb_epoch=10)
    evaluate_model(model)


def evaluate_model(classifier):
    y_pred = classifier.predict(X_validation)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_validation, y_pred)
    print(cm)
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    print(diagonal_sum / sum_of_all_elements)
