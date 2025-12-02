''' 
This file contains the functions for performing cross-validation.
'''

import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from data import load_data, impute_with_mode, cap_feature_values
from data_advanced import impute_with_median
from model import DecisionTree, init_ID3, MODEL_OPTIONS
from train import train, evaluate


def cross_validation(
    cv_folds: List[pd.DataFrame], id3_variant: str,
    depth_limit_values: List[int], num_trees: int = 25) -> Tuple[int, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of dataframes, each corresponding to a fold of the data
        depth_limit_values (list): a list of depth_limit hyperparameter values to try
        ig_criterion (str): the information gain variant to use. Should be one of "entropy" or "collision".

    Returns:
        int: the best depth_limit hyperparameter discovered
        float: the average cross-validation F1 corresponding to the best depth_limit
    '''

    best_depth = None
    best_avg_score = 0.0

    for depth in depth_limit_values:
        fold_scores = []

        for i in range(len(cv_folds)):
            test_df = cv_folds[i]
            train_folds = [cv_folds[j] for j in range(len(cv_folds)) if j != i]
            train_df = pd.concat(train_folds, ignore_index=True)

            train_y = train_df.iloc[:, -1].tolist()
            train_x = train_df.iloc[:, :-1]
            test_y  = test_df.iloc[:, -1].tolist()
            test_x  = test_df.iloc[:, :-1]

            # Cap/Bin values and Impute
            train_x = cap_feature_values(train_x)
            test_x  = cap_feature_values(test_x)

            if id3_variant == "simple":
                train_x_imputed, mode_vals = impute_with_mode(train_x)
                test_x_imputed, _ = impute_with_mode(test_x, mode_vals)
            else:
                train_x_imputed, mode_vals = impute_with_median(train_x)
                test_x_imputed, _ = impute_with_median(test_x, mode_vals)

            # SMOTE functionality, not really used but its here just in case
            # train_x, train_y = smote_balance(train_x, train_y)

            # Instantiate model
            if id3_variant == "simple":
                model = init_ID3("simple", depth_limit=depth)
            elif id3_variant == "bagging":
                model = init_ID3("bagging", depth_limit=depth, num_trees=num_trees)
            elif id3_variant == "randomforest":
                model = init_ID3("randomforest", depth_limit=depth, num_trees=num_trees)
            else:
                raise ValueError(f"Invalid variant: {id3_variant}. Must be one of {MODEL_OPTIONS}")

            # Train model
            train(model, train_x_imputed, train_y)

            # Evaluate (F1-macro)
            score = evaluate(model, test_x_imputed, test_y, type='f1macro')
            fold_scores.append(score)

        avg_score = sum(fold_scores) / len(fold_scores)
        print(f"Depth {depth}: CV F1-macro = {avg_score:.4f}")

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_depth = depth

    return best_depth, best_avg_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-validation for ID3 / Bagging / RandomForest')
    parser.add_argument("--cv_path", "-c", type=str, required=True,
        help="Path to CV fold directory")
    parser.add_argument("--model", "-m", type=str, default="simple", choices=MODEL_OPTIONS,
        help=f"ID3 variant to evaluate: {MODEL_OPTIONS}")
    parser.add_argument("--num_trees", "-nt", type=int, default=25,
        help="Number of trees (for bagging and random forest)")
    args = parser.parse_args()

    # load data using the provided path
    data_dict = load_data(args.cv_path)
    cv_folds = list(data_dict.values())
    
    depths_to_test = list(range(1, 9))

    print(f"Running CV for variant={args.model} on depths={depths_to_test}...")

    best_depth, best_f1 = cross_validation(
        cv_folds=cv_folds,
        id3_variant=args.model,
        depth_limit_values=depths_to_test,
        num_trees=args.num_trees
    )
    
    # Print final results
    print('-----')
    print(f"Best depth found: {best_depth}")
    print(f"Best average CV F1-macro: {best_f1:.4f}")