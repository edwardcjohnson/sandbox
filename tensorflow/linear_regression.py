import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_dateframe(start, stop, n_samples):
    # intialise dataframe
    df = pd.DataFrame({'x':np.linspace(start, stop, n_samples)} )
    # create response
    df['y'] = create_exponential_growth_with_noise(a=1, r=2, t=df['x'])
    return df

def create_exponential_growth_with_noise(a, r, t):
    e = np.random.normal(0, 1, len(t))
    return a * r * np.exp(t) + e

df = create_dataframe(0, 100, 101)
df.plot('x','y')    

# split the data into train and test sets
# this lets us simulate how our model will perform in the future
df_train, df_test = train_test_split(df, df['y'], test_size=0.33)
N, D = df_train.shape   


# Scale the data
# you'll learn why scaling is needed in a later course
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train['x'] = scaler.fit_transform(df_train['x'])
df_test['x'] = scaler.transform(df_test['x'])

# Now create the Tensorflow model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')
# model.compile(optimizer='adam', loss='mse')


# learning rate scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

X = df_train['x'].reshape(-1, 1) # make df['x'] a 2-D array of size N x D where D = 1
Y = df_train['y']
# Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])
