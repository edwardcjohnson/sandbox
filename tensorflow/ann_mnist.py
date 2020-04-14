import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

# read mnist data from tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# scale pixels from [0,255] to [0,1]
x_train, x_test = x_train / 255, x_test / 255

print(x_train.shape)

# render an mnist image

# construct model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax') # 10 output classes
    ])
    
model.compile(
    optimizer='adam',
    loss='categorical_cross_entropy',
    metrics=['accuracy'])
    
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test)
    epochs=10)
    
df_history = pd.DataFrame(history.history)
df_history.plot()

    
