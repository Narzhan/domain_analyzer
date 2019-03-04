import pandas
import numpy as np
import json
# import string
# from gensim.models import word2vec
import matplotlib.pyplot as plt
# from gensim.models import FastText
# import gensim, \
import pickle
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
# from numpy import asarray
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# nltk.download('wordnet')


dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
languages = ["en", "cs", "de", "es", "fr", "ja", "ru", "zh"]
# filter_languages = ["en", "de", "es", "fr", "ja", "ru", "zh"]
# pretrained_dataset = dataset.loc[dataset["language"].isin(languages)]
# customtrain_dataset = dataset.loc[~dataset["language"].isin(languages)]


# def preprocess(text):
#     # convert to list the input
#     tokens = [key.lower() for key in nltk.word_tokenize(text)]
#     words = [word for word in tokens if word.isalpha()]
#     return words


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
# print(int(seq_lengths.describe()["max"]))

# processed_docs = customtrain_dataset['text'].map(preprocess)
# processed_docs = pickle.load(open("splitted_docs.pkl", "rb"))
# pickle.dump(processed_docs, open("splitted_text/word_embedding/splitted_docs.pkl", "wb"))
feature_size = 300
window_context = 10
min_word_count = 5
sample = 1e-3

####### Fast Text
# fast_text = FastText(processed_docs, size=feature_size,
#                      window=window_context, min_count=min_word_count, sample=sample,
#                      iter=50)
# fast_text.save("splitted_text/word_embedding/fast_text/fast_text.pkl")
# fast_text.wv.save_word2vec_format("pretrained/custom_embedding.txt", binary=False)
# fast_text, processed_docs = None, None

######Word2Vec
# w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=10)
# w2v_model.save("w2v_model.pkl")
# w2v_model = word2vec.Word2Vec.load("splitted_text/word_embedding/w2v/w2v_custom_model.pkl")

# t = Tokenizer()
# t.fit_on_texts(dataset.text)
t = pickle.load(open("splitted_text/word_embedding/tokenizer.pkl", "rb"))
vocab_size = len(t.word_index) + 1
# embedding_matrix = np.zeros((vocab_size, 300))
# files_location = ["wiki.cs.vec", "wiki.de.vec", "wiki.en.vec", "wiki.es.vec", "wiki.fr.vec", "wiki.ja.vec",
#                   "wiki.ru.vec", "wiki.zh.vec"]
# for file_location in files_location:
#     with open("pretrained/{}".format(file_location), "r") as file:
#         next(file)
#         for line in file:
#             values = line.split()
#             if values[0] in t.word_index:
#                 # word = values[0]
#                 coefs = asarray(values[len(values)-300:], dtype='float32') # convert to float16
#                 embedding_matrix[t.word_index[values[0]]] = coefs
#
# with open("pretrained/custom_embedding.txt", "r") as file:
#     for line in file:
#         values = line.split()
#         # if values[0] in t.word_index:
#         # word = values[0]
#         coefs = asarray(values[len(values) - 300:], dtype='float32')
#         embedding_matrix[t.word_index[values[0]]] = coefs
# np.save("splitted_text/word_embedding/fast_text/embedding_martix.npy", embedding_matrix)

####Wod2vec
# embedding_matrix = np.zeros((len(w2v_model.wv.vocab), feature_size))
# for i in range(len(w2v_model.wv.vocab)):
#     embedding_matrix[t.word_index[values[0]]] = coefs
#
#     embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
# w2v_model = gensim.models.Word2Vec.load("de.bin")
# for word in w2v_model.wv.vocab:
#     if word in t.word_index:
#         embedding_vector = w2v_model.wv[word]
#         if embedding_vector is not None:
#             embedding_matrix[t.word_index[word]] = embedding_vector
# np.save("splitted_text/word_embedding/w2v_embedding_martix_custom.npy", embedding_matrix)
# from gensim.models import KeyedVectors
# # Load vectors directly from the file
# model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)


# X_train, X_test, y_train, y_test = train_test_split(dataset.text, dataset.label, test_size=0.1, random_state=7)
# X_train_seq = t.texts_to_sequences(X_train)
# X_test_seq = t.texts_to_sequences(X_test)
MAX_LEN = 134
X_train_seq_trunc = pad_sequences(t.texts_to_sequences(dataset.text), maxlen=MAX_LEN)
# X_test_seq_trunc = pad_sequences(t.texts_to_sequences(X_test), maxlen=MAX_LEN)
X_train, X_valid, y_train, y_valid, domains_train, domains_valid = train_test_split(X_train_seq_trunc, dataset.label,
                                                                                    dataset.index.tolist(),
                                                                                    test_size=0.2, random_state=7,
                                                                                    stratify=dataset.label)
X_train_seq_trunc = None
dataset, t = None, None
num_epoches = 6
batch = 8049
num_neurons = 50


def create_rnn_lstm(method: str):
    # Add an Input Layer
    # input_layer = Input((MAX_LEN,))
    #
    # # Add the word embedding Layer
    # embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
    #     input_layer)
    # embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    #
    # # Add the LSTM Layer
    # lstm_layer = LSTM(num_neurons)(embedding_layer)
    #
    # # Add the output Layers
    # output_layer1 = Dense(50, activation="relu")(lstm_layer)
    # output_layer1 = Dropout(0.25)(output_layer1)
    # output_layer2 = Dense(1, activation="sigmoid")(output_layer1)
    #
    # # Compile the model
    # model = Model(inputs=input_layer, outputs=output_layer2)
    # model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model = load_model("untrained/lstm_{}.h5".format(method))
    # Fitting our model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch, epochs=num_epoches)
    model.save("lstm_{}.h5".format(method))
    evaluate_model(model)
    with open("train_results_lstm_{}.json".format(method), "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "lstm")
    # eval_metric(history, "loss", "en", method, "lstm")
    # eval_metric(history, "acc", "cz", method, "lstm")
    # eval_metric(history, "loss", "cz", method, "lstm")


def create_rnn_gru(method: str):
    # Add an Input Layer
    # input_layer = Input((MAX_LEN,))
    #
    # # Add the word embedding Layer
    # embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
    #     input_layer)
    # embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    #
    # # Add the GRU Layer
    # lstm_layer = GRU(num_neurons)(embedding_layer)
    #
    # # Add the output Layers
    # output_layer1 = Dense(50, activation="relu")(lstm_layer)
    # output_layer1 = Dropout(0.25)(output_layer1)
    # output_layer2 = Dense(1, activation="sigmoid")(output_layer1)
    #
    # # Compile the model
    # model = Model(inputs=input_layer, outputs=output_layer2)
    # model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model = load_model("untrained/gru_{}.h5".format(method))
    # Fitting our model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch, epochs=num_epoches)
    model.save("gru_{}.h5".format(method))
    evaluate_model(model)
    with open("train_results_gru_{}.json".format(method), "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "gru")
    # eval_metric(history, "loss", "en", method, "gru")
    # eval_metric(history, "acc", "cz", method, "gru")
    # eval_metric(history, "loss", "cz", method, "gru")


def create_bidirectional_rnn(method: str):
    # Add an Input Layer
    # input_layer = Input((MAX_LEN,))
    #
    # # Add the word embedding Layer
    # embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
    #     input_layer)
    # embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    #
    # # Add the LSTM Layer
    # lstm_layer = Bidirectional(GRU(num_neurons))(embedding_layer)
    #
    # # Add the output Layers
    # output_layer1 = Dense(50, activation="relu")(lstm_layer)
    # output_layer1 = Dropout(0.25)(output_layer1)
    # output_layer2 = Dense(1, activation="sigmoid")(output_layer1)
    #
    # # Compile the model
    # model = Model(inputs=input_layer, outputs=output_layer2)
    # model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model = load_model("untrained/bilstm_{}.h5".format(method))
    # Fitting our model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch, epochs=num_epoches)
    model.save("bilstm_{}.h5".format(method))
    evaluate_model(model)
    with open("train_results_bilstm_{}.json".format(method), "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "bilstm")
    # eval_metric(history, "loss", "en", method, "bilstm")
    # eval_metric(history, "acc", "cz", method, "bilstm")
    # eval_metric(history, "loss", "cz", method, "bilstm")


def without_embedding():
    # Add an Input Layer
    # input_layer = Input((MAX_LEN,))
    #
    # # Add the word embedding Layer
    # embedding_layer = Embedding(vocab_size, 100)(input_layer)
    # embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    #
    # # Add the LSTM Layer
    # lstm_layer = LSTM(num_neurons)(embedding_layer)
    # pooling = GlobalMaxPool1D()(embedding_layer)
    # # Add the output Layers
    # output_layer1 = Dense(50, activation="relu")(lstm_layer)
    # output_layer1 = Dropout(0.25)(output_layer1)
    # output_layer2 = Dense(1, activation="sigmoid")(output_layer1)
    #
    # # Compile the model
    # model = Model(inputs=input_layer, outputs=output_layer2)
    # model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model = load_model("untrained/no_mbedding_lstm.h5")
    # Fitting our model
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch, epochs=num_epoches)
    model.save("no_mbedding_lstm.h5")
    evaluate_model(model)
    with open("train_results_no_embedding.json", "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "lstm")
    # eval_metric(history, "loss", "en", method, "lstm")
    # eval_metric(history, "acc", "cz", method, "lstm")
    # eval_metric(history, "loss", "cz", method, "lstm")


def eval_metric(history, metric_name, lang, emb, nn):
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
    try:
        plt.savefig("{}_{}_{}_{}.png".format(emb, nn, metric_name, lang))
    except Exception as e:
        print(e)


def evaluate_model(classifier):
    y_pred = classifier.predict(X_valid, batch_size=2048)
    y_pred = (y_pred > 0.5)
    print(accuracy_score(y_valid, y_pred))
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred))


def transform_data(model_name: str):
    model = load_model(model_name)
    x_train = model.predict(X_train, batch_size=1024)
    x_train = (x_train > 0.5)
    x_valid = model.predict(X_train, batch_size=1024)
    x_valid = (x_valid > 0.5)
    result_dataset = pandas.DataFrame(data=np.concatenate((x_train, x_valid), axis=0),
                                  index=np.concatenate((domains_train, domains_valid), axis=0), columns=["we"])
    result_dataset.to_csv("result_data.csv")
    # result_dataset = pandas.DataFrame({"predictions": np.concatenate((x_train, x_valid), axis=0),
    #                                    "domains": np.concatenate((domains_train, domains_valid), axis=0),
    #                                    "labels": np.concatenate((y_train, y_valid), axis=0)}
    #                                   )
    # result_dataset.to_csv("{}_temp.csv".format(model_name))
    # result_dataset = result_dataset.groupby(['domains', 'labels'])['predictions'].apply(
    #     lambda x: ','.join(x.astype(str))).reset_index()
    # result_dataset["predictions"] = [tuple(x.split(",")) for x in dataset["predictions"]]
    # temp_dataset = pandas.DataFrame(dataset["predictions"].values.tolist())
    # temp_dataset["labels"] = result_dataset["labels"]
    # temp_dataset["domains"] = result_dataset["domains"]
    # temp_dataset.set_index("domains", inplace=True)
    # temp_dataset.to_csv("{}.csv".format(model_name))

for model in ["bilstm_fasttext_mixed.h5", "lstm_w2v_custom.h5", "lstm_w2v_mixed.h5", "no_embedding_trained_bias_l1.h5"]:
    model = model.replace(".h5", "")
    transform_data(model)

# for emb_type in ["w2v_embedding_martix_mixed.npy", "fasttext_embedding_martix_custom.npy",
#                  "fasttext_embedding_martix_mixed.npy", "w2v_embedding_martix_custom.npy"]:
#     # embedding_matrix = np.load(emb_type)
#     method = emb_type.replace("_embedding_martix", "").replace(".npy", "")
#     print(method)
#     print("#####################")
#     if method == "w2v_custom":
#         create_rnn_lstm(method)
#         create_rnn_gru(method)
#     create_bidirectional_rnn(method)
# without_embedding()

# #Glove model
# model_glove = Sequential()
# model_glove.add(Embedding(vocabulary_size, feature_size, input_length=MAX_LEN, weights=[embedding_matrix], trainable=False))
# model_glove.add(Dropout(0.2))
# model_glove.add(Conv1D(64, 5, activation='relu'))
# model_glove.add(MaxPooling1D(pool_size=4))
# model_glove.add(LSTM(100))
# model_glove.add(Dense(1, activation='sigmoid'))
# model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


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