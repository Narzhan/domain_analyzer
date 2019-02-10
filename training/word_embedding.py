import pandas, nltk
import numpy as np
import string
from gensim.models import word2vec
import matplotlib.pyplot as plt
from gensim.models import FastText
import gensim, pickle
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
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


dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
languages = ["en", "cs", "de", "es", "fr", "ja", "ru", "zh"]
# filter_languages = ["en", "de", "es", "fr", "ja", "ru", "zh"]
# pretrained_dataset = dataset.loc[dataset["language"].isin(languages)]
customtrain_dataset = dataset.loc[~dataset["language"].isin(languages)]


def preprocess(text):
    # convert to list the input
    tokens = [key.lower() for key in nltk.word_tokenize(text)]
    words = [word for word in tokens if word.isalpha()]
    return words


# stemmer = SnowballStemmer('english')
#
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

# seq_lengths = dataset.text.apply(lambda x: len(x.split(' ')))
# print(seq_lengths.describe())

processed_docs = customtrain_dataset['text'].map(preprocess)
# processed_docs = pickle.load(open("splitted_docs.pkl", "rb"))
# pickle.dump(processed_docs, open("splitted_text/word_embedding/splitted_docs.pkl", "wb"))
feature_size = 300
window_context = 10
min_word_count = 5
sample = 1e-3


####### Fast Text
fast_text = FastText(processed_docs, size=feature_size,
                     window=window_context, min_count=min_word_count, sample=sample,
                     iter=50)
fast_text.save("splitted_text/word_embedding/fast_text/fast_text.pkl")
fast_text.wv.save_word2vec_format("pretrained/custom_embedding.txt", binary=False)
fast_text, processed_docs = None, None

######Word2Vec
# w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=10)
# w2v_model.save("w2v_model.pkl")


t = Tokenizer()
t.fit_on_texts(dataset.text)
vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 300))
files_location = ["wiki.cs.vec", "wiki.de.vec", "wiki.en.vec", "wiki.es.vec", "wiki.fr.vec", "wiki.ja.vec",
                  "wiki.ru.vec", "wiki.zh.vec"]
# files_location = ["wiki.cs.vec"]
for file_location in files_location:
    with open("pretrained/{}".format(file_location), "r") as file:
        next(file)
        for line in file:
            values = line.split()
            if values[0] in t.word_index:
                # word = values[0]
                coefs = asarray(values[len(values)-300:], dtype='float32')
                embedding_matrix[t.word_index[values[0]]] = coefs

with open("pretrained/custom_embedding.txt", "r") as file:
    for line in file:
        values = line.split()
        # if values[0] in t.word_index:
        # word = values[0]
        coefs = asarray(values[len(values) - 300:], dtype='float32')
        embedding_matrix[t.word_index[values[0]]] = coefs
np.save("splitted_text/word_embedding/fast_text/embedding_martix.npy", embedding_matrix)

####Wod2vec
# embedding_matrix = np.zeros((len(w2v_model.wv.vocab), feature_size))
# for i in range(len(w2v_model.wv.vocab)):
#     embedding_matrix[t.word_index[values[0]]] = coefs
#
#     embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


# X_train, X_test, y_train, y_test = train_test_split(dataset.text, dataset.label, test_size=0.1, random_state=7)
# X_train_seq = t.texts_to_sequences(X_train)
# X_test_seq = t.texts_to_sequences(X_test)
MAX_LEN = 134
X_train_seq_trunc = pad_sequences(t.texts_to_sequences(dataset.text), maxlen=MAX_LEN)
# X_test_seq_trunc = pad_sequences(t.texts_to_sequences(X_test), maxlen=MAX_LEN)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_seq_trunc, dataset.label, test_size=0.2,
                                                      random_state=7)
X_train_seq_trunc = None
dataset, t = None, None
num_epoches = 15
batch = 64
num_neurons = 100

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = Input((MAX_LEN,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = LSTM(num_neurons)(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    history = model.fit(X_train, y_train, batch_size=batch, nb_epoch=num_epoches)
    evaluate_model(model)
    eval_metric(history, "acc", "en")
    eval_metric(history, "loss", "en")
    eval_metric(history, "acc", "cz")
    eval_metric(history, "loss", "cz")


def create_rnn_gru():
    # Add an Input Layer
    input_layer = Input((MAX_LEN,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = GRU(num_neurons)(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    history = model.fit(X_train, y_train, batch_size=batch, epochs=num_epoches)
    evaluate_model(model)
    eval_metric(history, "acc", "en")
    eval_metric(history, "loss", "en")
    eval_metric(history, "acc", "cz")
    eval_metric(history, "loss", "cz")


def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = Input((MAX_LEN,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = Bidirectional(GRU(num_neurons))(embedding_layer)

    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    history = model.fit(X_train, y_train, batch_size=batch, nb_epoch=num_epoches)
    evaluate_model(model)
    eval_metric(history, "acc", "en")
    eval_metric(history, "loss", "en")
    eval_metric(history, "acc", "cz")
    eval_metric(history, "loss", "cz")


def eval_metric(history, metric_name, lang):
    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axisbatch
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, num_epoches + 1)

    if lang == "cz":
        train, valid = "Trénovací ", "Validační "
    else:
        train, valid = "Train ", "Validation "
    plt.plot(e, metric, 'bo', label=train + metric_name)
    plt.plot(e, val_metric, 'b', label=valid + metric_name)
    plt.legend()
    plt.show()


def evaluate_model(classifier):
    y_pred = classifier.predict(X_valid)

    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    print(diagonal_sum / sum_of_all_elements)

# #Glove model
# model_glove = Sequential()
# model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
# model_glove.add(Dropout(0.2))
# model_glove.add(Conv1D(64, 5, activation='relu'))
# model_glove.add(MaxPooling1D(pool_size=4))
# model_glove.add(LSTM(100))
# model_glove.add(Dense(1, activation='sigmoid'))
# model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])