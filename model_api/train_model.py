import argparse
import pandas as pd
import lightgbm as lgb
import numpy as np

def train_model(dataset_path: str, model_name: str):
    """Trains a LightGBM regression model on a synthetic dataset and saves it to disk.

    Args:
        dataset_path (str): Path to the synthetic dataset
        model_name (str): Name of the file to save the trained model

    Returns:
        None

    Raises:
        None

    Example:
        >>> train_model('synthetic_data.csv', 'model.txt')
    """
    # Load the synthetic dataset
    df = pd.read_csv(dataset_path)

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
    model.save_model(model_name)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a LightGBM regression model on a synthetic dataset')
    parser.add_argument('--dataset_path', type=str, default='synthetic_data.csv', help='Path to the synthetic dataset')
    parser.add_argument('--model_name', type=str, default='model.txt', help='Name of the file to save the trained model')
    args = parser.parse_args()

    # Train the model
    train_model(args.dataset_path, args.model_name)
