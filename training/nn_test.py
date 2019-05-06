import pandas as pd
import time

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Embedding, SpatialDropout1D, GlobalMaxPool1D, LSTM, Conv1D, Activation, Flatten, MaxPooling1D
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
# dataset= pd.read_csv("result.csv")
from tensorflow.python.keras import Input, Model


def load_test_data(cols: list):
    dataset = pd.read_csv("dataframe_enhanced.csv", index_col=0,
                          usecols=['part_path', 'fresh', 'pages', 'totalEstimatedMatches', "topics", "tf_idf",
                                   "embedding",
                                   'label', "domain"])
    # dataset = dataset.sort_index()
    #     # test_dataset = pd.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
    #     # test_dataset = test_dataset.replace(np.nan, '', regex=True)
    #     # test_dataset = test_dataset.sort_index()
    #     # tfidf = pickle.load(open("binaries/tfidf(0.1).pkl", "rb"))
    #     # features = tfidf.transform(test_dataset.text)
    # dataset_lda = pd.read_csv("splitted_text/lda/result_data.csv", index_col=0)
    # dataset = dataset.join(dataset_lda)
    # dataset_tfidf = pd.read_csv("splitted_text/tf_idf/result_data.csv", index_col=0)
    # dataset = dataset.join(dataset_tfidf)
    # col_list = list(dataset)
    # col_list[-3], col_list[-1] = col_list[-1], col_list[-3]
    # dataset.columns = col_list

    array = dataset.values
    X = array[:, 0:-1]
    Y = array[:, -1]
    # X = sparse.hstack([features, topics, w2v_labels, X])

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, dataset.index, test_size=0.2,
                                                                                     stratify=Y, random_state=7)
    return X_train, X_test, y_train, y_test, X.shape, indices_train, indices_test


def baseline_nn(X, X_validm, name):
    start_time = time.time()
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=5))
    classifier.add(Dropout(0.3))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=20, init='uniform', activation='relu'))
    classifier.add(Dropout(0.2))
    # Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # classifier.save("test.h5")
    # classifier = load_model("test.h5")
    # Fitting our model
    history = classifier.fit(X, Y_train, validation_data=(X_validm, Y_validation), batch_size=64, epochs=5)
    classifier.save("pca/{}_model_nn.h5".format(name), include_optimizer=False)
    print(history)
    evaluate_model(classifier, X_validm)
    score, acc = classifier.evaluate(X_validation, Y_validation, batch_size=32)
    print(score, acc)


def lstm_nn():
    # input_layer = Input((length[1],))

    # # Add the word embedding Layer
    # embedding_layer = Embedding(length[0]+1, length[1])(input_layer)
    # embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # # Add the LSTM Layer
    # lstm_layer = LSTM(50)(embedding_layer)
    # pooling = GlobalMaxPool1D()(embedding_layer)
    # # Add the output Layers
    # output_layer1 = Dense(50, activation="relu")(lstm_layer)
    # output_layer1 = Dropout(0.25)(output_layer1)
    # output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # # Compile the model
    # model = Model(inputs=input_layer, outputs=output_layer2)
    # model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    # # classifier.save("test.h5")
    # # classifier = load_model("test.h5")
    # # Fitting our model
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, nb_epoch=5)
    # evaluate_model(model)
    # score, acc = model.evaluate(X_test, y_test, batch_size=32)
    # print(score, acc)
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=20,
                      input_shape=(length[1], length[0]),
                      kernel_initializer= 'uniform',
                      activation= 'relu') )
    model.add(Activation('sigmoid'))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Conv1D(filters=128, kernel_size=20) )
    model.add(Activation('sigmoid'))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Conv1D(1, kernel_size=20,
                      kernel_initializer= 'uniform',
                      activation= 'relu') )
    model.add(Flatten() )
    model.add(Dense(128, kernel_initializer='normal', activation='relu') )
    model.add(Dense(1, activation='sigmoid', name='output') )

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=64, nb_epoch=5)
    evaluate_model(model)
    score, acc = model.evaluate(X_validation, Y_validation, batch_size=32)
    print(score, acc)


def evaluate_model(classifier, x_val):
    y_pred = classifier.predict(x_val, batch_size=32)
    y_pred = (y_pred > 0.5)
    print(accuracy_score(Y_validation, y_pred))
    print(confusion_matrix(Y_validation, y_pred))
    print(classification_report(Y_validation, y_pred))
    # predictions = map(lambda x: 1 if x else 0, y_pred)
    # with open("splitted_text/fp_fn_nn.txt", "w") as file:
    #     counter = 0
    #     for input, prediction, label in zip(indices_test, predictions, y_test):
    #         if prediction != label:
    #             file.write(
    #                 "Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
    #                                                                                        list(X_test[
    #                                                                                                 counter])))
    #         counter += 1


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

    e = range(1, 5 + 1)

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

columns = ['full_path', 'part_path', 'about',
               'deep_links', 'fresh', 'infection', 'pages', 'totalEstimatedMatches', 'someResultsRemoved', 'label',
               "domain"]
X_train, X_validation, Y_train, Y_validation, length, indices_train, indices_test = load_test_data(columns)
# from sklearn.preprocessing import MinMaxScaler
# scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_validation = scaling.transform(X_validation)
import pickle
for name in ["standard"]:
    scaler = pickle.load(open("pca/{}.pkl".format(name), "rb"))
    X_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_validation)
    pca = pickle.load(open("pca/pca.pkl", "rb"))
    X_scaled = pca.transform(X_scaled)
    X_valid_scaled = pca.transform(X_valid_scaled)
    baseline_nn(X_scaled, X_valid_scaled, name)
