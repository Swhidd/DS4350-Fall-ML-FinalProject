''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse
import numpy as np
from data import load_data
from evaluate import accuracy, f1_macro
from model import init_perceptron, MajorityBaseline, MODEL_OPTIONS

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    train_x = data_dict['train_x']
    train_y = data_dict['train_y']
    test_x  = data_dict['test_x']
    eval_ids = data_dict['eval_ids']
    print("Test std deviation per feature:", np.std(test_x, axis=0))
    print("Number of near-constant features:", np.sum(np.std(test_x, axis=0) < 1e-6))

    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    print(f'  test x shape: {test_x.shape}')

    # Load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    if args.model == 'majority_baseline':
        model = MajorityBaseline()

        # Train the model
        print(f'train MajorityBaseline')
        model.train(x=train_x, y=train_y)
    
    else:
        model = init_perceptron(
            variant=args.model, 
            num_features=train_x.shape[1], 
            lr=args.lr, 
            mu=args.mu)
        print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

        # Train the model
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

    # Evaluate model on train and test data
    print('evaluate')
    train_predictions = model.predict(x=train_x)
    train_f1 = f1_macro(train_y, train_predictions)
    print(f'  train f1: {train_f1:.3f}')
    print(f'  Number of mistakes: {model.get_mistakes()}')
    
    # Get predictions for submission
    test_predictions = model.predict(test_x)

    # Create submission
    submission_path = 'data/perceptron_submission.csv'
    with open(submission_path, 'w', encoding='utf-8') as f:
        f.write('example_id,label\n')
        for ex_id, pred_label in zip(eval_ids, test_predictions):
            f.write(f'{ex_id},{pred_label}\n')

    print(f'Submission file written to: {submission_path}')
