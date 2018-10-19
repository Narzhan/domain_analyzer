import numpy as np # linear algebra
import pandas as pd
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

dataset= pd.read_csv("result.csv")

array = dataset.values
X = array[:, 0:12]
Y = array[:, 12]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=12))
# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting our model
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=10)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)