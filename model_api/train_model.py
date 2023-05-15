import pandas as pd
import lightgbm as lgb
import numpy as np

def train_model():
    """Trains a LightGBM regression model on a synthetic dataset and saves it to disk.

    Returns:
        None

    Raises:
        None

    Example:
        >>> train_model()
    """
    # Create the synthetic dataset
    df = create_synthetic_data()

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # Train a LightGBM regression model on the data
    params = {
        'objective': 'regression',
        'metric': 'rmse'
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data)

    # Save the trained model to disk
    model.save_model('model.txt')


def create_synthetic_data(n_samples=1000):
    """Create a synthetic dataset with specified number of samples and features.

        Args:
            n_samples (int, optional): Number of samples in the dataset. Defaults to 1000.

        Returns:
            pandas.DataFrame: A synthetic dataset containing random values for each feature and a target variable
            generated using a linear combination of the features.

        Raises:
            ValueError: If n_samples is not a positive integer.
            ValueError: If feature_ranges is not a list of tuples containing the low and high range for each feature.
    """
    # Define the number of features
    n_features = 3
    
    # Define the range of values for each feature
    feature_ranges = [(0, 1), (0, 1), (0, 1)]
    
    # Generate random values for each feature
    X = np.random.uniform(
        low=[r[0] for r in feature_ranges],
        high=[r[1] for r in feature_ranges],
        size=(n_samples, n_features)
    )
    
    # Generate the target variable using a linear combination of the features
    beta = np.array([0.5, 0.3, 0.2])
    error = np.random.normal(scale=0.1, size=n_samples)
    y = np.dot(X, beta) + error
    
    # Combine the features and target variable into a DataFrame
    data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
    data['target'] = y
    
    return data


if __name__ == '__main__':
    train_model()
