import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

def create_dateframe(min, max, n_samples):
    # intialise dataframe
    df = pd.DataFrame({
        'x1': (max - min) * random_sample(n_samples) + min
        'x2': (max - min) * random_sample(n_samples) + min
        } )
    # create response
    df['y'] = create_nonlinear_response(df)
    return df

def create_nonlinear_response(df):
    return np.cos(2 * df['x1']) + np.cos(3 * df['x2'])

df = create_dataframe(-3, 3, 1000)

# 3-D plot of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x1'], df['x2'], df['y'])
# plt.show()

# split the data into train and test sets
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.33)

X = ['x1', 'x2']
y = 'y'

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile the model
opt = tf.keras.optimizers.Adam(0.01) # adam opt with custom learning rate
model.compile(optimizer=opt, loss='mse')
history = model.fit(
    df_train[X],  # .to_numpy()
    df_train[y],
    validation_data=(df_test[X], df_test[y])
)

df_history = pd.DataFrame(history.history)
for metric in ['loss', 'accuracy']:
    df_history[metric, f'val_{metric}'].plot()

    
def plot_prediction_surface(df, min, max):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x1'], df['x2'], df['y'])
    # surface plot
    line = np.linspace(min, max, 50)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    Yhat = model.predict(Xgrid).flatten()
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
    plt.show()

plot_prediction_surface(df, model, -3, 3)
# Can the model extrapolate?
# Answer: No. The activation function is not periodic, 
#   so model cannot predict a periodic function like this one.
plot_prediction_surface(df, model, -10, 10)
