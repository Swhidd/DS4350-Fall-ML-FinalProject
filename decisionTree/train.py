''' This file contains the functions for training and evaluating a model.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
import numpy as np
import pandas as pd
from data import load_data, impute_with_mode, cap_feature_values
from data_advanced import impute_with_median
from model import DecisionTree, MajorityBaseline, Model
from generate_confusion_matrix import confusion_matrix_binary, print_confusion_matrix
# from bagging_id3 import BaggingClassifier
# from random_forest_id3 import RandomForestClassifier

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from model import MODEL_OPTIONS, init_ID3


def train(model: Model, x: pd.DataFrame, y: list):
    '''
    Learn a model from training data.
    '''

    trainedModel = model.train(x, y)
    return trainedModel


def evaluate(model: Model, x: pd.DataFrame, y: list, type: str) -> float:
    '''
    Evaluate a trained model against a dataset.
    Type can be 'accuracy' or 'f1macro' depending on the desired test.
    '''
    
    predictions = model.predict(x)
    
    if type == 'accuracy':
        output = calculate_accuracy(y, predictions)
    if type == 'f1macro':
        output = calculate_f1_macro_manual(y, predictions)
    return output

def calculate_accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    '''
    correctPredictions = 0
    for i in range(len(labels)):
        # print("pred @ i", predictions[i], "\n label @ i", labels[i])
        if labels[i] == predictions[i]:
            correctPredictions += 1
    
    accuracy = safe_div(correctPredictions, len(labels))
    return accuracy

def safe_div(n: float, d: float) -> float:
    """Return n/d, or 0.0 if d == 0 (safe divide)."""
    return n / d if d else 0.0

def create_submission(model: Model, x: pd.DataFrame):
    predictions = model.predict(x)
    for i in range(len(predictions)):
        predictions[i] = int(predictions[i])
    submissionDf = pd.DataFrame({
        'example_id': range(0, len(predictions)),
        'label': predictions
    })
    submissionDf.to_csv("data/id3_submission.csv", index=False)

def smote_balance(xTrain, yTrain, ratio=0.33):
    smote = SMOTE(sampling_strategy=ratio, random_state=42)
    xSmote, ySmote = smote.fit_resample(xTrain, yTrain)
    return xSmote, ySmote

def calculate_f1_macro_manual(labels, predictions) -> float:
    '''
    Manual implementation of F1-macro score calculation.
    '''
    if len(labels) == 0 or len(predictions) == 0:
        return 0.0
    
    # Get all unique classes
    classes = set(labels) | set(predictions)
    
    f1_scores = []
    
    for class_label in classes:
        # Calculate TP, FP, FN for current class
        tp = sum(1 for true, pred in zip(labels, predictions) if true == class_label and pred == class_label)
        fp = sum(1 for true, pred in zip(labels, predictions) if true != class_label and pred == class_label)
        fn = sum(1 for true, pred in zip(labels, predictions) if true == class_label and pred != class_label)
        
        # Calculate precision and recall
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        
        # Calculate F1-score for this class
        if precision + recall == 0:
            f1_class = 0.0
        else:
            f1_class = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1_class)
    
    # F1-macro is the average of F1-scores for all classes
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return f1_macro

def confusion_matrix_binary(labels: list, predictions: list):
    """
    Compute confusion matrix for binary classes [0,1].
    Returns (TN, FP, FN, TP).
    """
    TN = FP = FN = TP = 0

    for true, pred in zip(labels, predictions):
        if true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1
        elif true == 1 and pred == 1:
            TP += 1

    return TN, FP, FN, TP


def print_confusion_matrix(TN, FP, FN, TP):
    """
    Prints a confusion matrix.
    """
    print("\nConfusion Matrix")
    print("                Predicted 0    Predicted 1")
    print(f"Actual 0       |    {TN:5d}         {FP:5d}")
    print(f"Actual 1       |    {FN:5d}         {TP:5d}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ID3 model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument("-t", "--train_path", type=str, required=True)
    parser.add_argument("-e", "--eval_path", type=str, required=True)
    parser.add_argument("-i", "--ids_path", type=str, required=True)
    parser.add_argument("-dl", "--depth_limit", type=int, default=None)
    parser.add_argument("-nt", "--num_trees", type=int, default=25)
    args = parser.parse_args()

    # Load training set
    train_df = pd.read_csv(args.train_path)
    train_x = train_df.drop(train_df.columns[-1], axis=1)
    train_y = train_df[train_df.columns[-1]].tolist()

    # Cap/Bin values and Impute
    train_x = cap_feature_values(train_x)

    if args.model == 'simple':
        train_x, mode_vals = impute_with_mode(train_x)
    else:
        train_x, median_vals = impute_with_median(train_x)

    # SMOTE functionality, not really used but its here just in case
    # train_x, train_y = smote_balance(train_x, train_y, ratio=0.33)

    # Instantiate model
    print(f'initialize model')
    if args.model == 'majority_baseline':
        model = MajorityBaseline()
    else:
        model = init_ID3(
            variant=args.model, 
            depth_limit=args.depth_limit, 
            num_trees=args.num_trees)
        print(f'  model type: {type(model).__name__}')

    # Train model
    train(model, train_x, train_y)

    # Load testing set
    eval_df = pd.read_csv(args.eval_path)
    test_x = eval_df

    # Cap/Bin values and Impute
    test_x = cap_feature_values(test_x)

    if args.model == 'simple':
        test_x, _ = impute_with_mode(test_x)
    else:
        test_x, _ = impute_with_median(test_x)

    # Get model predictions
    preds = model.predict(test_x)
    preds = [int(p) for p in preds]

    print(f"Confusion matrix for training set with {args.model} ID3.")
    TN, FP, FN, TP = confusion_matrix_binary(train_y, model.predict(train_x))
    print_confusion_matrix(TN,FP,FN,TP)

    # Load .ids for creating submission
    ids = [int(line.strip()) for line in open(args.ids_path)]

    # Create submission
    sub = pd.DataFrame({
        "example_id": ids,
        "label": preds
    })

    sub.to_csv("data/submission.csv", index=False)
    print("Saved submission.csv")