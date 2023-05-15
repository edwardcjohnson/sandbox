import pandas as pd
import numpy as np
import argparse


def create_synthetic_data(n_samples=1000, feature_ranges=[(0, 1), (0, 1), (0, 1)]):
    """Create a synthetic dataset with specified number of samples and features.

        Args:
            n_samples (int, optional): Number of samples in the dataset. Defaults to 1000.
            feature_ranges (list of tuples, optional): Low and high range for each feature. Defaults to [(0, 1), (0, 1), (0, 1)].

        Returns:
            pandas.DataFrame: A synthetic dataset containing random values for each feature and a target variable
            generated using a linear combination of the features.

        Raises:
            ValueError: If n_samples is not a positive integer.
            ValueError: If feature_ranges is not a list of tuples containing the low and high range for each feature.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(feature_ranges, list) or not all(isinstance(r, tuple) and len(r) == 2 for r in feature_ranges):
        raise ValueError("feature_ranges must be a list of tuples containing the low and high range for each feature")

    # Define the number of features
    n_features = len(feature_ranges)
    
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


def main(n_samples, feature_ranges, output_file):
    data = create_synthetic_data(n_samples, feature_ranges)
    data.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='Number of samples in the dataset')
    parser.add_argument('-r', '--feature_ranges', nargs='+', type=float, default=[(0, 1), (0, 1), (0, 1)], help='Low and high range for each feature')
    parser.add_argument('-o', '--output_file', type=str, default='data.csv', help='Output file path')
    args = parser.parse_args()

    feature_ranges = [(float(r[0]), float(r[1])) for r in zip(args.feature_ranges[::2], args.feature_ranges[1::2])]
    main(args.n_samples, feature_ranges, args.output_file)
