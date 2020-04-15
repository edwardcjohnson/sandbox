import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# read in the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)

# the data is only 2D!
# convolution expects height x width x color
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes
k = len(set(y_train))
print(f"number of classes: {k}")

# Build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Compile and fit
# Note: GPU will help a lot here
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

df_history = pd.DataFrame(history.history)
for metric in ['loss', 'accuracy']:
    df_history[metric, f'val_{metric}'].plot()
    
def plot_confusion_matrix(y, y_pred):
    df_confusion_matrix = pd.DataFrame(sklearn.metrics.confusion_matrix(y, y_pred))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_confusion_matrix, annot=True, annot_kws={"size": 16}) # font size

y_test_pred = model.predict(y_test)
plot_confusion_matrix(y_test, y_test_pred)
    
def show_misclassified_example(y, y_pred):
    y_misclassified = y[y != y_pred]
    i = np.random.choice(y_misclassified)
    plt.imshow(y_misclassified[i], cmap='gray')
    
show_misclassified_example(y_test, y_test_pred)
