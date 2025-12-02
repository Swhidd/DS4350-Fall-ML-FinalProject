''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd

def impute_with_median(df: pd.DataFrame, median_values=None):
    """
    Impute missing values using column medians.
    This is safer for sparse numeric malware features than mode imputation.
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

    # Load and impute training set
    train_raw = pd.read_csv('data/train_original.csv')
    train_x_df = train_raw.drop('label', axis=1)
    train_y = train_raw['label']

    train_x_df, median_vals = impute_with_median(train_x_df)

    # Load and impute test set
    test_raw = pd.read_csv('data/test_original.csv')

    test_x_df, _ = impute_with_median(test_raw, median_vals)

    with open('data/eval.ids', 'r', encoding='utf-8') as f:
        eval_ids = [line.strip() for line in f.readlines()]

    # Load cross validation datasets
    cv_folds = []
    for cv_fold_path in glob('data/cv/*'):
        fold_raw = pd.read_csv(cv_fold_path)

        fold_x = fold_raw.drop('label', axis=1)
        fold_y = fold_raw['label']

        # Impute fold using median values from training
        fold_x_imputed, _ = impute_with_median(fold_x, median_vals)

        # Reconstruct fold DataFrame
        fold_imputed = pd.concat([fold_y, fold_x_imputed], axis=1)
        cv_folds.append(fold_imputed)

    return {
        'train_x': train_x_df.to_numpy(),
        'train_y': train_y.to_numpy(),
        'test_x': test_x_df.to_numpy(),
        'eval_ids': eval_ids,
        'cv_folds': cv_folds}
