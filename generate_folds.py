import os
import numpy as np
import pandas as pd


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with only a single unique value.
    This is safe as a global preprocessing step since such columns
    cannot provide information gain for any model.
    """
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]

    if len(constant_cols) > 0:
        print(f"Dropping {len(constant_cols)} constant columns:")
        print(constant_cols)
        df = df.drop(columns=constant_cols)
    else:
        print("No constant columns found.")

    return df


def make_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """
    Create stratified cross-validation folds without sklearn.
    This ensures each fold has approximately the same label distribution.
    """
    np.random.seed(seed)

    label_col = df.columns[-1]
    y = df[label_col]

    # Separate by class label
    grouped = {label: df[df[label_col] == label] for label in y.unique()}

    # Shuffle each class subset
    for label in grouped:
        grouped[label] = grouped[label].sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split each class subset into n_splits partitions
    class_slices = {
        label: np.array_split(grouped[label], n_splits)
        for label in grouped
    }

    folds = []
    for i in range(n_splits):
        fold_parts = [class_slices[label][i] for label in class_slices]
        fold = pd.concat(fold_parts, ignore_index=True)

        fold = fold.sample(frac=1, random_state=seed).reset_index(drop=True)
        folds.append(fold)

    return folds


def save_folds(folds, output_dir: str):
    """
    Save each fold to a CSV file named fold_1.csv, ..., fold_5.csv.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, fold in enumerate(folds, start=1):
        path = os.path.join(output_dir, f"fold_{idx}.csv")
        fold.to_csv(path, index=False)
        print(f"Saved {path} ({len(fold)} rows)")


if __name__ == "__main__":
    input_path = "data/train_original.csv"
    output_dir = "data/cv"

    print("Loading training data...")
    df = pd.read_csv(input_path)
    print(f"Loaded training data with shape: {df.shape}")

    df = drop_constant_columns(df)
    print(f"Shape after dropping constant columns: {df.shape}")

    print("Creating stratified 5-fold CV splits...")
    folds = make_stratified_folds(df, n_splits=5)

    print("Saving folds...")
    save_folds(folds, output_dir)

    print("Done generating CV folds.")
