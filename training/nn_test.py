import numpy as np # linear algebra
import pandas as pd
import time
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

    array = dataset.values
    X = array[:, 0:len(cols) - 2]
    Y = array[:, len(cols) - 2]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, y_train, y_test, len(cols)


def lear_nn(cols: list):
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    X_train, X_test, y_train, y_test, length = load_data(cols)
    start_time = time.time()
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=length - 2))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting our model
    classifier.fit(X_train, y_train, batch_size=100, nb_epoch=10)
    print(cols)
    print("result: {}".format(time.time() - start_time))
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    print(diagonal_sum / sum_of_all_elements)


if __name__ == '__main__':
    columns = ["ranking_response", 'full_path', 'part_path', 'about',
               'deep_links', 'fresh', 'infection', 'pages', 'totalEstimatedMatches', 'someResultsRemoved', 'label',
               "domain"]
    for column in ['full_path', 'about', 'deep_links', 'fresh', 'infection', 'someResultsRemoved']:
        columns.remove(column)
        lear_nn(columns)
