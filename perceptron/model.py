''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol, Tuple
from collections import Counter

import numpy as np

# set the numpy random seed so our randomness is reproducible
np.random.seed(1)


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the Perceptron classes further down.
class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...


class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        self.mostCommonLabel = ""
        self.predictions = []


    def get_hyperparams(self) -> dict:
        return {}
    

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''
    
        commonLabel = Counter(y).most_common(1)
        self.mostCommonLabel = commonLabel[0][0]
        print(f"Most common label: {self.mostCommonLabel}")

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        predictions = [self.mostCommonLabel] * len(x)
        self.predictions = predictions
        return self.predictions


class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''     

        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu
        self.num_features = num_features

        self.weightVector = []
        self.bias = None
        self.mistakes = 0

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}
    
    def get_mistakes(self) -> int:
        return self.mistakes
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # Need to convert labels from [0,1] to [-1,+1]
        y = np.where(y == 0, -1, 1)

        # Init weight vector with slight variance around 0 for learning rate to function.
        self.weightVector = np.random.uniform(-0.01, 0.01, self.num_features)
        self.bias = np.random.uniform(-0.01, 0.01)

        t = 0

        for epoch in range(epochs):
            x, y = shuffle_data(x, y)

            for i in range(len(x)):
                activation = np.dot(self.weightVector, x[i]) + self.bias

                # Margin
                if self.mu is not None and self.mu > 0:
                    # if prediction != y[i]:
                    if activation * y[i] < self.mu:
                        if self.decay_lr:
                            lr = self.lr / (1 + t)
                        else:
                            lr = self.lr
                        self.weightVector += lr * y[i] * x[i]
                        self.bias += lr* y[i]
                        t += 1
                        self.mistakes += 1
                # Decay
                elif self.decay_lr:
                    # if prediction != y[i]:
                    if activation * y[i] < 0:
                        lr = self.lr / (1 + t)
                        self.weightVector += lr * y[i] * x[i]
                        self.bias += lr * y[i]
                        t += 1
                        self.mistakes += 1
                # Simple
                else:
                    if activation * y[i] < 0:
                        self.weightVector += self.lr * y[i] * x[i]
                        self.bias += self.lr * y[i]
                        self.mistakes += 1

        return self.weightVector
                
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        predictions = np.sign(np.dot(x, self.weightVector) + self.bias)

        # Resolve sign(0) = +1
        predictions[predictions == 0] = 1

        # Convert labels back to [0,1] from [-1,+1] for submission creation
        predictions = np.where(predictions == -1, 0, 1)

        return predictions.tolist()
    
    def train_one_epoch(self, x, y):
        return self.train(x, y, epochs=1)
    

class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate eta
        '''     

        self.lr = lr
        self.num_features = num_features

        self.weightVector = []
        self.bias = None
        self.averageWeightVector = []
        self.averageBias = None
        self.mistakes = 0
        

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}
    
    def get_mistakes(self) -> int:
        return self.mistakes
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''
        y = np.where(y == 0, -1, 1)

        self.weightVector = np.random.uniform(-0.01, 0.01, self.num_features)
        self.bias = np.random.uniform(-0.01, 0.01)
        self.averageWeightVector = np.zeros(self.num_features)
        self.averageBias = 0

        for epoch in range(epochs):
            x, y = shuffle_data(x, y)
            for i in range(len(x)):
                activation = np.dot(self.weightVector, x[i]) + self.bias
                if activation * y[i] < 0:
                    self.weightVector += self.lr * y[i] * x[i]
                    self.bias += self.lr * y[i]
                    self.mistakes += 1
                self.averageWeightVector += self.weightVector
                self.averageBias += self.bias
        normalizationFactor = epochs * len(x)
        self.averageWeightVector = self.averageWeightVector / normalizationFactor
        self.averageBias = self.averageBias / normalizationFactor
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        predictions = np.sign(np.dot(x, self.averageWeightVector) + self.averageBias)

        # Resolve sign(0) = +1
        predictions[predictions == 0] = 1

        # Convert labels back to [0,1] from [-1,+1] for submission creation
        predictions = np.where(predictions == -1, 0, 1)
        print("average weigth l2 norm:", np.linalg.norm(self.averageWeightVector))
        print("average bias:", self.averageBias)
        return predictions.tolist()
    

    def train_one_epoch(self, x, y):
        return self.train(x, y, epochs=1)
    

class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        
        # YOUR CODE HERE


    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
    

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        return []


# DON'T MODIFY THE FUNCTIONS BELOW!
PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS
def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]
