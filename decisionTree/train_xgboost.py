import xgboost as xgb
import numpy as np
import pandas as pd
from data_advanced import load_data

def main():
    # Load train and test data
    data_dict = load_data()

    train_x = data_dict['train_x']
    train_y = data_dict['train_y']
    test_x  = data_dict['test_x']
    eval_ids = data_dict['eval_ids']

    # Instantiate XGB Model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss'
    )

    model.fit(train_x, train_y)

    preds = model.predict(test_x)

    preds = preds.astype(int)

    print("Writing submission file...")

    with open("data/submission_xgboost.csv", "w", encoding='utf-8') as f:
        f.write("example_id,label\n")
        for ex_id, label in zip(eval_ids, preds):
            f.write(f"{ex_id},{label}\n")

    print("Saved submission_xgboost.csv")

if __name__ == "__main__":
    main()
