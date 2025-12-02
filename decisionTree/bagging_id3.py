import numpy as np
import pandas as pd
from model import DecisionTree


class BaggingClassifier:
    """
    Bootstrap Aggregation w/ ID3. Each tree trains on bootstrapped samples with all features.
    """

    def __init__(self, depth_limit=8, num_trees=25, ig_criterion='entropy'):
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.trees = []

    def train(self, x: pd.DataFrame, y: list):
        n = len(x)
        for i in range(self.num_trees):

            # Sample rows with replacement
            bootstrap_idx = np.random.choice(n, size=n, replace=True)
            x_boot = x.iloc[bootstrap_idx]
            y_boot = [y[j] for j in bootstrap_idx]

            # Train the tree
            tree = DecisionTree(depth_limit=self.depth_limit, ig_criterion=self.ig_criterion)
            tree.train(x_boot, y_boot)

            self.trees.append(tree)

    def predict(self, x: pd.DataFrame):

        # Collect predictions from all trees
        all_preds = np.array([tree.predict(x) for tree in self.trees])

        # Majority vote across columns
        votes = np.sum(all_preds, axis=0)
        final_preds = np.where(votes >= (self.num_trees / 2), 1, 0)
        return final_preds.tolist()