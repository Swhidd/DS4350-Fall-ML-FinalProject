''' This file contains functions for evaluating model predictions.
    You don't need to change this file.
'''

def accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    assert len(labels) == len(predictions), (
        f'{len(labels)=} and {len(predictions)=} must be the same length.' + 
        '\n\n  Have you implemented model.predict()?\n'
    )
    
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    accuracy = correct / len(labels)

    return accuracy

def f1_macro(labels: list, predictions: list) -> float:
    """
    Manual F1-macro implementation (same as ID3).
    """
    if len(labels) == 0 or len(predictions) == 0:
        return 0.0

    classes = set(labels) | set(predictions)
    f1_scores = []

    for c in classes:
        tp = sum(1 for y, p in zip(labels, predictions) if y == c and p == c)
        fp = sum(1 for y, p in zip(labels, predictions) if y != c and p == c)
        fn = sum(1 for y, p in zip(labels, predictions) if y == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)