''' This file contains the code for finding the optimal training epochs.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import load_data, impute_with_median
from evaluate import accuracy, f1_macro
from model import init_perceptron, Model, PERCEPTRON_VARIANTS


def train_epochs(
        model: Model, 
        train_x: np.ndarray, 
        train_y: np.ndarray, 
        val_x: np.ndarray, 
        val_y: np.ndarray,
        epochs: int = 20) -> Tuple[int, float]:
    '''
    Run epoch training to find the ideal number of epochs.

    Args:
        model (Model): the perceptron model to train
        train_x (np.ndarray): a numpy ndarray containing the training features
        train_y (np.ndarray): a numpy ndarray containing the training labels
        val_x (np.ndarray): a numpy ndarray containing the validation features
        val_y (np.ndarray): a numpy ndarray containing the validation labels
        epochs (int): the maximum number of epochs to train for. Defaults to 20

    Returns:
        int: the optimal number of epochs
        float: the validation accuracy corresponding to the optimal number of epochs
    '''
    
    best_epochs = -1
    best_val_accuracy = 0
    valAccuracyList = []
    for epoch in range(1, epochs + 1):
        # print("First five lines of training data:")
        # for i in range(5):
        #     print(train_x[i])
        model.train_one_epoch(train_x, train_y)
        # testPredictions = model.predict(val_x)
        # valAccuracy = accuracy(val_y, testPredictions)
        # print(f"Epoch: {epoch}, valAccuracy: {valAccuracy}, Current best_val_accuracy: {best_val_accuracy}")
        # valAccuracyList.append(valAccuracy)
        # if valAccuracy > best_val_accuracy:
        #     best_val_accuracy = valAccuracy
        #     best_epochs = epoch
        predictions = model.predict(val_x)

        # ---------------------------------------
        # USE F1-MACRO FOR EPOCH SELECTION
        # ---------------------------------------
        score = f1_macro(val_y, predictions)

        print(f"Epoch {epoch}: val F1-macro = {score:.4f} (best = {best_val_accuracy:.4f})")

        valAccuracyList.append(score)
        if score > best_val_accuracy:
            best_val_accuracy = score
            best_epochs = epoch
    make_epoch_chart(epochs, valAccuracyList)
    return best_epochs, best_val_accuracy

def make_epoch_chart(epochs, valAccuracy):
    epochList = [i for i in range(1, epochs + 1)]
    plt.plot(epochList, valAccuracy, marker='o')
    plt.xlabel("Epochs")
    plt.xticks(epochList)
    plt.ylabel("Validation Set Accuracy")
    plt.title("Change in Validation Set Accuracy over the Epochs")
    plt.show()




# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=PERCEPTRON_VARIANTS, 
        help=f'Which perceptron model to run. Must be one of {PERCEPTRON_VARIANTS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='How many epochs to train for. Defaults to 20.')
    args = parser.parse_args()

    # Load data
    print('load data')
    data_dict = load_data()
    full_x = data_dict['train_x']
    full_y = data_dict['train_y']

    # Don't have validation set, simulate one with simple internal split.
    # 90% train, 10% validation
    n = len(full_x)
    split = int(n * 0.9)

    train_x = full_x[:split]
    train_y = full_y[:split]
    val_x   = full_x[split:]
    val_y   = full_y[split:]

    # Impute train and validation sets
    train_x_df = pd.DataFrame(train_x)
    val_x_df   = pd.DataFrame(val_x)
    train_x_imputed, median_vals = impute_with_median(train_x_df)
    val_x_imputed, _ = impute_with_median(val_x_df, median_vals)
    train_x = train_x_imputed.to_numpy()
    val_x   = val_x_imputed.to_numpy()

    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    print(f'  val x shape: {val_x.shape}\n  val y shape: {val_y.shape}')

    # Load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    model = init_perceptron(
        variant=args.model, 
        num_features=train_x.shape[1], 
        lr=args.lr, 
        mu=args.mu)
    print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

    # Train the model to find optimal number of epochs
    best_epochs, best_accuracy = train_epochs(
        model=model, 
        train_x=train_x, 
        train_y=train_y, 
        val_x=val_x, 
        val_y=val_y,
        epochs=args.epochs)
    print(
        f'\noptimal number of epochs from epoch training' +
        f':\n\n         epochs: {best_epochs:5d}\n       accuracy: {best_accuracy:.3f}\n')
