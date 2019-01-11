import numpy as np # linear algebra
import pandas as pd
import time, pickle

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K


# dataset= pd.read_csv("result.csv")
def load_data(cols: list):
    dataset = pd.read_csv("test_data.csv", index_col=len(cols) - 1, usecols=cols)
    dataset = dataset.sort_index()
    test_dataset = pd.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
    test_dataset = test_dataset.replace(np.nan, '', regex=True)
    test_dataset = test_dataset.sort_index()
    tfidf = pickle.load(open("binaries/tfidf(0.1).pkl", "rb"))
    features = tfidf.transform(test_dataset.text)
    test_dataset, tfidf = None, None
    topics = pickle.load(open("gensim_lda/topics_25.pkl", "rb"))
    w2v_labels = pickle.load(open("cluster_labels.pkl", "rb")).reshape((-1, 1))

    array = dataset.values
    X = array[:, 0:len(cols) - 2]
    Y = array[:, len(cols) - 2]
    X = sparse.hstack([features, topics, w2v_labels, X])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, y_train, y_test, X.shape[1]


def baseline_nn():
    start_time = time.time()
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=length))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    classifier.fit(X_train, y_train, batch_size=100, nb_epoch=10)
    print("result: {}".format(time.time() - start_time))
    evaluate_model(classifier)


def evaluate_model(classifier):
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    print(diagonal_sum / sum_of_all_elements)


columns = ['full_path', 'part_path', 'about',
               'deep_links', 'fresh', 'infection', 'pages', 'totalEstimatedMatches', 'someResultsRemoved', 'label',
               "domain"]
X_train, X_test, y_train, y_test, length = load_data(columns)
baseline_nn()
