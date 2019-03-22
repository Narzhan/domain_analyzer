import pandas as pd
import time

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Embedding, SpatialDropout1D, GlobalMaxPool1D, LSTM
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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


def baseline_nn():
    start_time = time.time()
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=length[1]))
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
    history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=5)
    classifier.save("dense_model.h5", include_optimizer=False)
    evaluate_model(classifier)
    score, acc = classifier.evaluate(X_test, y_test, batch_size=32)
    print(score, acc)


def lstm_nn():
    input_layer = Input((length[1],))

    # Add the word embedding Layer
    embedding_layer = Embedding(length[0]+1, length[1])(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = LSTM(50)(embedding_layer)
    pooling = GlobalMaxPool1D()(embedding_layer)
    # Add the output Layers
    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    # classifier.save("test.h5")
    # classifier = load_model("test.h5")
    # Fitting our model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, nb_epoch=5)
    evaluate_model(model)
    score, acc = model.evaluate(X_test, y_test, batch_size=32)
    print(score, acc)


def evaluate_model(classifier):
    y_pred = classifier.predict(X_test, batch_size=32)
    y_pred = (y_pred > 0.5)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    predictions = map(lambda x: 1 if x else 0, y_pred)
    with open("splitted_text/fp_fn_nn.txt", "w") as file:
        counter = 0
        for input, prediction, label in zip(indices_test, predictions, y_test):
            if prediction != label:
                file.write(
                    "Domain {} with incorrect label: {}, should be: {}, data: {}\n".format(input, prediction, label,
                                                                                           list(X_test[
                                                                                                    counter])))
            counter += 1


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
X_train, X_test, y_train, y_test, length, indices_train, indices_test = load_test_data(columns)
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
baseline_nn()