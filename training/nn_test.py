import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
# Add another:
model.add(keras.layers.Dense(64, activation='sigmoid'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))
# Or:
# layers.Dense(64, activation=tf.sigmoid)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

filename = "result.csv"
record_defaults = [tf.int64] * 13
dataset = tf.contrib.data.CsvDataset(filename, record_defaults, header=True)
dataset=dataset.batch(32)
dataset=dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30)

# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
# dataset = dataset.batch(32).repeat()
#
# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
# val_dataset = val_dataset.batch(32).repeat()
#
# model.fit(dataset, epochs=10, steps_per_epoch=30,
#           validation_data=val_dataset,
#           validation_steps=3)