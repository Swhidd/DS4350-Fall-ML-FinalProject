import numpy as np
import pandas as pd
from model import DecisionTree


class RandomForestClassifier:
    """
    Random Forest w/ ID3. 
    Uses per-tree feature sampling as this was easier to implement with existing ID3 code.
    Sample features f = (sqrt(d) * 3) b/c f = sqrt(d) gave poor performance due to low amount of features.
    """
    def __init__(self, depth_limit=10, num_trees=50, max_features=None, ig_criterion='entropy'):
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.max_features = max_features
        self.ig_criterion = ig_criterion
        self.trees = []

    def train(self, x: pd.DataFrame, y: list):
        n, d = x.shape
        cols = list(x.columns)

        if self.max_features is None:
            self.max_features = int(np.sqrt(d))

        for i in range(self.num_trees):

            # Bootstrap sample rows
            bootstrap_idx = np.random.choice(n, size=n, replace=True)
            x_boot = x.iloc[bootstrap_idx]
            y_boot = [y[j] for j in bootstrap_idx]

            # Train on ID3 tree that performs random feature subsampling at each node via feature_subsample_size=self.max_features
            tree = DecisionTree(depth_limit=self.depth_limit, ig_criterion=self.ig_criterion)
            tree.train(x_boot, y_boot, feature_subsample_size=self.max_features)

            self.trees.append(tree)

    def predict(self, x: pd.DataFrame):
        preds = np.array([tree.predict(x) for tree in self.trees])
        votes = np.sum(preds, axis=0)
        final_preds = np.where(votes >= (self.num_trees / 2), 1, 0)
        return final_preds.tolist()
