import numpy as np
import pandas as pd
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
