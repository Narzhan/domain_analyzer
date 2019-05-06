import pandas
import numpy as np
import json
# import string
# from gensim.models import word2vec
import matplotlib.pyplot as plt
from gensim.models import FastText
import gensim
import pickle
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Dropout, GRU, Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
nltk.download('wordnet')


dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
languages = ["en", "cs", "de", "fr", "ru"]
# # filter_languages = ["en", "de", "es", "fr", "ja", "ru", "zh"]
# # pretrained_dataset = dataset.loc[dataset["language"].isin(languages)]
# customtrain_dataset = dataset.loc[~dataset["language"].isin(languages)]
# dataset = None

# # # def preprocess(text):
# # #     # convert to list the input
# # #     tokens = [key.lower() for key in nltk.word_tokenize(text)]
# # #     words = [word for word in tokens if word.isalpha()]
# # #     return words


# stemmer = SnowballStemmer('english')


# def lemmatize_stemming(text):
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# def preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             result.append(lemmatize_stemming(token))
#     return result

# # # seq_lengths = dataset.text.apply(lambda x: len(x.split(' ')))
# # # print(int(seq_lengths.describe()["max"]))

# processed_docs = customtrain_dataset['text'].map(preprocess)
# # # print(processed_docs)

# # # processed_docs = pickle.load(open("splitted_docs.pkl", "rb"))
# # # pickle.dump(processed_docs, open("splitted_text/word_embedding/splitted_docs.pkl", "wb"))
feature_size = 300
# min_word_count = 2
# window_context = 10
# sample = 1e-3

# ###### Fast Text
# fast_text = FastText(processed_docs, size=feature_size,
#                      window=window_context, min_count=min_word_count, sample=sample,
#                      iter=10, sg=1, hs=1, workers=6, word_ngrams=1, min_n=2, max_n=20, sorted_vocab=1, negative=5)
# # fast_text.save("splitted_text/word_embedding/fast_text/fast_text.pkl")
# fast_text.wv.save_word2vec_format("pretrained/custom_embedding_al.txt", binary=False)
# fast_text, processed_docs = None, None

######Word2Vec
# w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=10)
# w2v_model.save("w2v_model.pkl")
# w2v_model = word2vec.Word2Vec.load("splitted_text/word_embedding/w2v/w2v_custom_model.pkl")

# t = Tokenizer()
# t.fit_on_texts(dataset.text)
t = pickle.load(open("tokenizer.pkl", "rb"))
vocab_size = len(t.word_index) + 1
# embedding_matrix = np.zeros((vocab_size, 300))
# #embedding_matrix = np.load("fasttext_embedding_martix_mixed.npy")
# files_location = ["wiki.cs.align.vec", "wiki.de.align.vec", "wiki.en.align.vec", "wiki.fr.align.vec",
#                   "wiki.ru.align.vec"]
# for file_location in files_location:
#     with open("aligned/{}".format(file_location), "r") as file:
#         next(file)
#         for line in file:
#             values = line.split()
#             if values[0] in t.word_index:
#                 # word = values[0]
#                 coefs = asarray(values[len(values)-300:], dtype='float32') # convert to float16
#                 embedding_matrix[t.word_index[values[0]]] = coefs
#
# with open("pretrained/custom_embedding_al.txt", "r") as file:
#     next(file)
#     for line in file:
#         values = line.split()
#         # if values[0] in t.word_index:
#         # word = values[0]
#         coefs = asarray(values[len(values) - 300:], dtype='float32')
#         try:
#             embedding_matrix[t.word_index[values[0]]] = coefs
#         except Exception as e:
#             pass
        
# np.save("pretrained/embedding_martix_custom_al.npy", embedding_matrix)
# print("Zeroes: {}".format(np.sum(~embedding_matrix.any(1))))

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
X_train, X_validation, Y_train, Y_validation, domains_train, domains_valid = train_test_split(X_train_seq_trunc, dataset.label,
                                                                                              dataset.index.tolist(),
                                                                                              test_size=0.2, random_state=7,
                                                                                              stratify=dataset.label)
X_train_seq_trunc = None
# dataset, t = None, None
t=None
num_epoches = 6
batch = 8049
num_neurons = 50
store_training = False


def create_rnn_lstm(method: str):
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
    # model = load_model("untrained/lstm_{}.h5".format(method))
    # Fitting our model
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=batch, epochs=num_epoches)
    model.save("lstm_{}.h5".format(method), include_optimizer=store_training)
    evaluate_model(model)
    with open("train_results_lstm_{}.json".format(method), "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "lstm")
    # eval_metric(history, "loss", "en", method, "lstm")
    # eval_metric(history, "acc", "cz", method, "lstm")
    # eval_metric(history, "loss", "cz", method, "lstm")


def create_rnn_gru(method: str):
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
    # model = load_model("untrained/gru_{}.h5".format(method))
    # Fitting our model
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=batch, epochs=num_epoches)
    model.save("gru_{}.h5".format(method), include_optimizer=store_training)
    evaluate_model(model)
    with open("train_results_gru_{}.json".format(method), "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "gru")
    # eval_metric(history, "loss", "en", method, "gru")
    # eval_metric(history, "acc", "cz", method, "gru")
    # eval_metric(history, "loss", "cz", method, "gru")


def create_bidirectional_rnn(method: str):
    # Add an Input Layer
    input_layer = Input((MAX_LEN,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, feature_size, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = Bidirectional(LSTM(num_neurons))(embedding_layer)
    pooling = GlobalMaxPool1D()(embedding_layer)    

    # Add the output Layers
    output_layer1 = Dropout(0.3)(lstm_layer)
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.3)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.save("bilstm_{}.h5".format(method), include_optimizer=store_training)

    # model = load_model("untrained/bilstm_{}.h5".format(method))
    # Fitting our model
    # history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=batch, epochs=num_epoches)
    # model.save("bilstm_{}.h5".format(method), include_optimizer=store_training)
    # evaluate_model(model)
    # model.load_weights()
    # with open("train_results_bilstm_{}.json".format(method), "w") as file:
    #     json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "bilstm")
    # eval_metric(history, "loss", "en", method, "bilstm")
    # eval_metric(history, "acc", "cz", method, "bilstm")
    # eval_metric(history, "loss", "cz", method, "bilstm")


def without_embedding():
    # Add an Input Layer
    input_layer = Input((MAX_LEN,))

    # Add the word embedding Layer
    embedding_layer = Embedding(vocab_size, 100)(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = LSTM(num_neurons)(embedding_layer)
    pooling = GlobalMaxPool1D()(embedding_layer)
    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    # model = load_model("untrained/no_mbedding_lstm.h5")
    # Fitting our model
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=batch, epochs=num_epoches)
    model.save("no_mbedding_lstm.h5", include_optimizer=store_training)
    evaluate_model(model)
    with open("train_results_no_embedding.json", "w") as file:
        json.dump(history.history, file)
    # eval_metric(history, "acc", "en", method, "lstm")
    # eval_metric(history, "loss", "en", method, "lstm")
    # eval_metric(history, "acc", "cz", method, "lstm")
    # eval_metric(history, "loss", "cz", method, "lstm")


def eval_metric(history, metric_name, lang, emb, nn):
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
    y_pred = classifier.predict(X_validation, batch_size=2048)
    y_pred = (y_pred > 0.5)
    print(accuracy_score(Y_validation, y_pred))
    print(confusion_matrix(Y_validation, y_pred))
    print(classification_report(Y_validation, y_pred))


def transform_data(model_name: str):
    model = load_model(model_name)
    x_train = model.predict(X_train, batch_size=1024)
    # x_train = (x_train > 0.5)
    x_valid = model.predict(X_validation, batch_size=1024)
    # x_valid = (x_valid > 0.5)
    # result_dataset = pandas.DataFrame(data=np.concatenate((x_train, x_valid), axis=0),
    #                               index=np.concatenate((domains_train, domains_valid), axis=0), columns=["we"])
    result_dataset = pandas.DataFrame({"domain": np.concatenate((domains_train, domains_valid), axis=0),
                                       "we": np.concatenate((x_train, x_valid), axis=0)})
    result_dataset.to_csv("backup.csv")
    result_dataset = result_dataset.groupby('domain').agg(lambda x: x.tolist())
    final_dataset = pandas.DataFrame(result_dataset['we'].values.tolist(), index=result_dataset.index)
    final_dataset = final_dataset.replace(np.nan, 0, regex=True)
    final_dataset.to_csv("train_data.csv")
    final_dataset["label"] = dataset["label"]
    final_dataset.to_csv("train_data2.csv")
    array = final_dataset.values
    X = array[:, 0:-1]
    Y = array[:, -1]
    X_trai, X_validatio, Y_train, Y_validation = model_selection.train_test_split(X, Y,
                                                                                    test_size=0.2,
                                                                                    random_state=7)
    classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifier.fit(X_trai, Y_train)
    pickle.dump(classifier, open("prob_model.pkl", "wb"))
    predictions = classifier.predict(X_validatio)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
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

# for model in ["embedding_martix.npy", "embedding_martix_custom.npy", "embedding_martix_custom_al.npy"]:
#     embedding_matrix = np.load("pretrained/{}".format(model))
#     model = model.replace(".npy", "")
#     create_bidirectional_rnn(model)
transform_data("we_model.h5")
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