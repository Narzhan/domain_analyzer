from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle, gensim
from gensim.models import word2vec
import pandas
import numpy as np

dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
processed_docs = pickle.load(open("gensim_lda/processed_docs.pkl", "rb"))
feature_size = 100  # Word vector dimensionality
window_context = 20  # Context window size

w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
                              window=window_context, iter=50)
X = w2v_model[w2v_model.wv.vocab]

model_lstm = Sequential()
model_lstm.add(Embedding(len(w2v_model.wv.vocab), feature_size, input_length=feature_size, trainable=False))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X, dataset.label, validation_split=0.2, epochs=10)
