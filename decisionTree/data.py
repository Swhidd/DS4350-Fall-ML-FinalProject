''' 
This file provides utility functions for loading data that you may find useful.
'''

from glob import glob
import os

import pandas as pd


def impute_with_mode(df: pd.DataFrame, mode_values=None):
    """
    Impute missing values by column mode.
    If mode_values is None, compute from df.
    Returns (imputed_df, mode_values).
    """
    if mode_values is None:
        mode_values = df.mode().iloc[0]

    df_imputed = df.fillna(mode_values)
    return df_imputed, mode_values

def cap_feature_values(df):
    df = df.copy()
    for col in df.columns:

        # Only numeric columns should be capped
        if pd.api.types.is_numeric_dtype(df[col]):

            # If there are too many distinct values, bin them into 5 buckets
            if df[col].nunique() > 10:
                df[col] = pd.cut(df[col], bins=5, labels=False)
    return df

def load_data(path: str) -> dict:
    '''
    Loads all the data required for this assignment from a given path.

    Args:
        path (str): the path to the data directory (e.g., 'data/cv/')

    Returns:
        dict: a dictionary containing the dataframes from the specified path
    '''

    data_dict = {}
    
    # Use the provided path to find and load the cross-validation fold files
    for file_path in glob(os.path.join(path, '*.csv')):
        fold_name = os.path.splitext(os.path.basename(file_path))[0]
        fold_df = pd.read_csv(file_path)
        data_dict[fold_name] = fold_df

    return data_dict