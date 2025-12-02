''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol
from collections import Counter

import pandas as pd
import numpy as np


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        self.commonLabel = ''
        self.predictions = []


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        commonLabel = Counter(y).most_common(1)
        self.commonLabel = commonLabel[0][0]
        # for col in x.columns:
        #     print("Col:",col,"\n", Counter(x[col].values).most_common())
        print("Most common label: ",commonLabel)
        # print(Counter(y).most_common(), "\n len of y", len(y))
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        predictions = [self.commonLabel] * len(x)
        self.predictions = predictions
        return predictions


class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.builtTree = None
        self.commonLabel = None


    def train(self, x: pd.DataFrame, y: list, feature_subsample_size=None):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''
        ySeries = pd.Series(y, index=x.index)
        attributes = x.columns
        commonLabel = Counter(y).most_common(1)
        self.commonLabel = commonLabel[0][0]
        output = ID3(x, attributes, ySeries, self.depth_limit, feature_subsample_size=feature_subsample_size)
        self.builtTree = output

    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        predictions = []
        for i, row in x.iterrows():
            predictions.append(predictSingle(self.builtTree, row, self.commonLabel))
        return predictions

def predictSingle(builtTree, rowOfx, commonLabel):
    if builtTree.isLeafNode():
        return builtTree.value
    else:
        featureSplitOn = builtTree.featureToSplitOn
        featureValue = rowOfx[featureSplitOn]
        if featureValue in builtTree.children:
            childNode = builtTree.children[featureValue]
            return predictSingle(childNode, rowOfx, commonLabel)
        else:
            return commonLabel

def ID3(x, Attributes, y:pd.Series, maxDepth=None, currentDepth=0, feature_subsample_size=None):
    '''
    Build decision tree using ID3
    Args:
        x: Set of examples
        Attributes: Set of measured attributes
    '''
    if (y.nunique() == 1):
        return TreeNode(value=y.iloc[0])
    elif (len(Attributes) == 0 or len(x) == 0):
        return TreeNode(value=y.mode()[0])
    elif maxDepth is not None and currentDepth >= maxDepth:
        return TreeNode(value=y.mode()[0])
    else:
        if feature_subsample_size is not None and len(Attributes) > feature_subsample_size:
            attrs_array = np.array(Attributes)
            sampled_indices = np.random.choice(len(attrs_array), size=feature_subsample_size, replace=False)
            candidate_attrs = attrs_array[sampled_indices]
        else:
            candidate_attrs = Attributes
        a = bestAttribute(x, candidate_attrs, y)

        if a is None:
            return TreeNode(value=y.mode()[0])
        
        rootNode = TreeNode(featureToSplitOn=a)
        
        for v in x[a].unique():
            subsetMask = x[a] == v
            S_v = x[subsetMask]
            y_v = y[subsetMask]
            if S_v.empty:
                rootNode.children[v] = TreeNode(value=y.mode()[0])
            else:
                remainingAttributes = Attributes[Attributes != a]
                rootNode.children[v] = ID3(S_v, remainingAttributes, y_v, maxDepth=maxDepth,
                                            currentDepth=currentDepth + 1, 
                                            feature_subsample_size=feature_subsample_size)
        return rootNode
    
def bestAttribute(x:pd.DataFrame, Attributes, y:pd.Series):
    # Set to -1 so that values with very small infogain can still be selected.
    highestInformationGain = -1
    bestAttribute = None
    for attribute in Attributes:
        currentInformationGain = informationGain(x, attribute, y)
        if currentInformationGain >= highestInformationGain:
            highestInformationGain = currentInformationGain
            bestAttribute = attribute
    return bestAttribute

def entropy(labels):
    counterList = Counter(labels).most_common()
    labelSize = len(labels)
    totalEntropy = 0
    for label in counterList:
        labelCount = label[1]
        classProportion = labelCount / labelSize
        if classProportion > 0:
            totalEntropy += - (classProportion * np.log2(classProportion))
    return totalEntropy
    
def informationGain(x:pd.DataFrame, featureToSplitOn, labels: pd.Series):
    totalEntropy = entropy(labels)
    featuresOfColumn = x[featureToSplitOn].unique()
    totalSubsetEntropy = 0
    labelSize = len(labels)
    for colFeature in featuresOfColumn:
        featureSubsetIndices = x[x[featureToSplitOn] == colFeature].index
        featureSubsetLabels = labels.loc[featureSubsetIndices]
        featureSubsetSize = len(featureSubsetLabels)
        featureSubsetEntropy = entropy(featureSubsetLabels)
        colFeatureWeight = featureSubsetSize / labelSize
        totalSubsetEntropy += colFeatureWeight * featureSubsetEntropy
    return totalEntropy - totalSubsetEntropy


class TreeNode:
    def __init__(self, featureToSplitOn=None, children=None, value=None):
        self.featureToSplitOn = featureToSplitOn
        self.children = children or {}
        self.value = value

    def addChild(self, featureValue, childNode):
        self.children[featureValue] = childNode

    def isLeafNode(self):
        return self.value is not None


ID3_VARIANTS = ['simple', 'bagging', 'randomforest']
MODEL_OPTIONS = ['majority_baseline'] + ID3_VARIANTS
def init_ID3(variant: str, depth_limit: int, num_trees: int) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the ID3 Decision Tree

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in ID3_VARIANTS, f'{variant=} must be one of {ID3_VARIANTS}'

    if variant == 'simple':
        return DecisionTree(depth_limit=depth_limit)
    elif variant == 'bagging':
        from bagging_id3 import BaggingClassifier
        return BaggingClassifier(depth_limit=depth_limit, num_trees=num_trees)
    elif variant == 'randomforest':
        from random_forest_id3 import RandomForestClassifier
        return RandomForestClassifier(depth_limit=depth_limit, num_trees=num_trees)
