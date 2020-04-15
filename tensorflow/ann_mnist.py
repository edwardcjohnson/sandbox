import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf

# read mnist data from tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# scale pixels from [0,255] to [0,1]
x_train, x_test = x_train / 255, x_test / 255

print(x_train.shape)

# render an mnist image
plt.imshow(x_train[0], cmap='gray')

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
