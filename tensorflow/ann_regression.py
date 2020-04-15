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

df = create_dataframe(-3, 3, 100)

# 3-D plot of data

# split the data into train and test sets
df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.33)

X = ['x1', 'x2']
y = 'y'
