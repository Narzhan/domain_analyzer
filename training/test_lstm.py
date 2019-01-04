from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle, gensim
from gensim.models import word2vec
import pandas
import numpy as np


def sentence_vectorizer(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    for word in words:
        if word in vocabulary:
            feature_vector = np.add(feature_vector, model[word])
        print(feature_vector)
    return feature_vector


def corpus_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [sentence_vectorizer(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


# dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
# dataset = dataset.replace(np.nan, '', regex=True)
# dataset = dataset.sort_index()
processed_docs = pickle.load(open("gensim_lda/processed_docs_splitted.pkl", "rb"))
feature_size = 100  # Word vector dimensionality
window_context = 20  # Context window size

# w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
#                               window=window_context, iter=50)
w2v_model = word2vec.Word2Vec.load("gensim_we/w2v_model_splitted.pkl")
# X = w2v_model[w2v_model.wv.vocab]
w2v_feature_array = corpus_vectorizer(corpus=processed_docs, model=w2v_model, num_features=feature_size)
w2v_model, processed_docs = None, None
dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()

model_lstm = Sequential()
model_lstm.add(Embedding(len(w2v_feature_array), feature_size, input_length=feature_size, trainable=False))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(w2v_feature_array, dataset.label, validation_split=0.2, epochs=10)
