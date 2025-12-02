''' This file contains the functions for performing cross-validation.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import itertools
from typing import List, Tuple

import pandas as pd
import numpy as np

from data import load_data, impute_with_median
from evaluate import accuracy, f1_macro
from model import init_perceptron, PERCEPTRON_VARIANTS


def cross_validation(
        cv_folds: List[pd.DataFrame], 
        perceptron_variant: str, 
        lr_values: list, 
        mu_values: list,
        epochs: int = 10) -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of pandas DataFrames, corresponding to folds of the data. 
            The first column of each DataFrame, called "label", corresponds to y, 
            while the remaining columns are the features x.
        perceptron_variant (str): The variant of the perceptron algorithm to use
        lr_values (list): a list of learning rate hyperparameter values to try
        mu_values (list): a list of margin (mu) hyperparameter values to try
        epochs (int): how many epochs to train each model for. Defaults to 10

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hints:
        - We've provided a helper function `init_perceptron()` in model.py to initialize your model. 
          You can call it with `model = init_perceptron(perceptron_variant, num_features=num_features, lr=lr, mu=mu)`
        - The python `itertools.product()` function returns the Cartesian product of multiple lists.
          You can call `itertools.product(lr_values, mu_values) to get all combinations as (lr, mu) tuples.
        - You can convert a pandas DataFrame to a numpy ndarray with `df.to_numpy()`
    '''

    best_hyperparams = {'lr': None, 'mu': None}
    best_avg_accuracy = 0
    hyperparamCombos = itertools.product(lr_values, mu_values)

    for lr, mu in hyperparamCombos:
        foldAccuracyList = []
        
        for i in range(len(cv_folds)):
            testFold = cv_folds[i]
            trainDf = pd.concat([fold for j, fold in enumerate(cv_folds) if j != i], ignore_index=True)

            # Create Train and Test sets
            train_y = trainDf.iloc[:, 0]
            train_x = trainDf.iloc[:, 1:]
            test_y  = testFold.iloc[:, 0]
            test_x  = testFold.iloc[:, 1:]

            # Impute values
            train_x_imputed, median_vals = impute_with_median(train_x)
            test_x_imputed, _ = impute_with_median(test_x, median_vals)

            # Convert to numpy arrays
            trainX = train_x_imputed.to_numpy()
            trainY = train_y.to_numpy()
            testX  = test_x_imputed.to_numpy()
            testY  = test_y.to_numpy()

            model = init_perceptron(variant=perceptron_variant, 
                                    num_features=trainX.shape[1], 
                                    lr=lr, mu=mu)
            model.train(x=trainX, y=trainY, epochs=epochs)
            testPredictions = model.predict(testX)
            
            # Evaluate with f1-macro
            foldScore = f1_macro(testY, testPredictions)
            foldAccuracyList.append(foldScore)

        averageFoldAccuracy = sum(foldAccuracyList) / len(foldAccuracyList)
        print(f"LR={lr}, MU={mu}, VAR={perceptron_variant}: F1-macro = {averageFoldAccuracy:.4f}")

        if averageFoldAccuracy > best_avg_accuracy:
            best_avg_accuracy = averageFoldAccuracy
            best_hyperparams = {'lr':lr, 'mu':mu}

    return best_hyperparams, best_avg_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-validation for different hyperparameters')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=PERCEPTRON_VARIANTS, 
        help=f'Which perceptron model to run. Must be one of {PERCEPTRON_VARIANTS}.')
    parser.add_argument('--lr_values', nargs='+', type=float, default=[1], 
        help='A list (space separated) of learning rate (eta) values to try. This is the same as the initial learning rate.')
    parser.add_argument('--mu_values', nargs='+', type=float, default=[0], 
        help='A list (space separated) of margin (mu) values to try. Defaults to [0].')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # Load data
    print('load data')
    data_dict = load_data()
    cv_folds = data_dict['cv_folds']

    # Run cross_validation
    print(f'run cross-validation')
    best_hyperparams, best_accuracy = cross_validation(
        cv_folds=cv_folds, 
        perceptron_variant=args.model,
        lr_values=args.lr_values, 
        mu_values=args.mu_values, 
        epochs=args.epochs)
    
    # Print best hyperparameters and accuracy
    print('\nbest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\n       accuracy: {best_accuracy:.3f}\n')
