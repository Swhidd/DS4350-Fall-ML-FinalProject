''' 
This file provides utility functions for loading data that you may find useful.
'''

from glob import glob

import pandas as pd

def impute_with_median(df: pd.DataFrame, median_values=None):
    """
    Impute missing values using column medians.
    Use median over other imputation methods due to variance in training data.
    """
    if median_values is None:
        median_values = df.median(numeric_only=True)

    df_imputed = df.fillna(median_values)
    return df_imputed, median_values


def load_data() -> dict:
    '''
    Loads all the data required for this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cv data as (x, y) tuples of np.ndarray matrices
    '''

    # Load training set
    train_raw = pd.read_csv('data/train_original.csv')
    train_x_df = train_raw.drop('label', axis=1)
    train_y = train_raw['label']

    # Impute train_x using its own column medians
    train_x_df, median_vals = impute_with_median(train_x_df)

    # Load test set and prepare eval_ids
    test_raw = pd.read_csv('data/test_original.csv')

    # Impute test_x using TRAIN'S median values
    test_x_df, _ = impute_with_median(test_raw, median_vals)

    with open('data/eval.ids', 'r', encoding='utf-8') as f:
        eval_ids = [line.strip() for line in f.readlines()]

    # Load cross validation datasets
    cv_folds = []
    for cv_fold_path in glob('data/cv/*'):

        fold_raw = pd.read_csv(cv_fold_path)

        # Split fold into x and y, then impute x using the training medians
        fold_x = fold_raw.drop('label', axis=1)
        fold_y = fold_raw['label']
        fold_x_imputed, _ = impute_with_median(fold_x, median_vals)

        # Reconstruct the fold DataFrame
        fold_imputed = pd.concat([fold_y, fold_x_imputed], axis=1)
        cv_folds.append(fold_imputed)

    return {
        'train_x': train_x_df.to_numpy(),
        'train_y': train_y.to_numpy(),
        'test_x': test_x_df.to_numpy(),
        'eval_ids': eval_ids,
        'cv_folds': cv_folds}
