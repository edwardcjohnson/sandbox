import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model

# read in the data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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

#******************************************
#   Model (fit without data augmentation)
#******************************************
i = Input(shape=x_train[0].shape)
# double the num of feature maps with each conv layer:
# do NOT use strided conv bc normal conv+maxpool works better here. Ref: VGG network (multiple conv before pooling)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
## conv group 1
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i) # need padding='same' to avoid shrinking image after each conv
x = BatchNormalization()(x) # BatchNormalization acts as a regularizer since mu and sigma change w/ each batch
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
# x = Dropout(0.2)(x) # Dropout didn't help with this dataset, so leave out
## conv group 2
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
# x = Dropout(0.2)(x) # Dropout didn't help with this dataset, so leave out
## conv group 3
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
# x = Dropout(0.2)(x) # Dropout didn't help with this dataset, so leave out
## Dense group
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(k, activation='softmax')(x)

model = Model(i, x)

# Compile
# Note: GPU will help a lot here
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Fit
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)
df_history = pd.DataFrame(history.history)
for metric in ['loss', 'accuracy']:
    df_history[metric, f'val_{metric}'].plot()

    
#******************************************
#   Model (fit with data augmentation)
#******************************************
# Note: if you run this AFTER calling the previous model.fit(), it will CONTINUE training where it left off.
# Usually that's what you want, but for experimentation we can re-compile the model to start from scratch.
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

history = model.fit(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=50)
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


model.summary() # show the model architecture
