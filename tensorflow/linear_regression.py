import numpy as np
import pandas as pd
import sklearn
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
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.33)

X = ['x']
y = 'y'

# Scale the data
scaler = sklearn.preprocessing.StandardScaler()
df_train[X] = scaler.fit_transform(df_train[X])
df_test[X] = scaler.transform(df_test[X])

# Now create the Tensorflow model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, input_shape=(len(X), )) 
])
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.001, 0.9),
    loss='mse')

# learning rate scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
history = model.fit(df_train[X], df_train[y], epochs=200, callbacks=[scheduler]) # .reshape(-1, 1)

df_history = pd.DataFrame(history.history)
for metric in ['loss', 'accuracy']:
    df_history[metric, f'val_{metric}'].plot()

