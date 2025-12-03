# **Android Malware Detection – Project README**

This repository contains two complete machine-learning pipelines for classifying Android applications as **benign (0)** or **malicious (1)** using a version of the **TUANDROMD dataset**.  

There are **two independent implementations** relevant for the project:

1. **Decision Tree–based models**  
   - ID3  
   - Bagging (Bootstrap Aggregation)  
   - Random Forest (feature subsampling)  
   - XGBoost baseline  
2. **Perceptron-based models**  
   - Averaged Perceptron  
   - Includes all the other Perceptron variants from the assignment, with the exception of Aggressive Perceptron.

Both pipelines implement:
- Median/mode imputation  
- Cross-validation without leakage  
- Manual F1-macro scoring  
- Submission file generation compatible with `eval.ids`

---

# **Repository Structure**

```
.
├── generate_folds/               <- utility for generating CV folds
├── decisionTree/                 <- decision-tree implementation
│   ├── data/
│   │   └── (submission files appear here when generated, this can be changed in the train.py files)
│   ├── bagging_id3.py
│   ├── cross_validation.py
│   ├── data.py
│   ├── data_advanced.py
│   ├── model.py
│   ├── random_forest_id3.py
│   ├── train.py
│   ├── train_xgboost.py
├── perceptron/                   <- perceptron implementation
│   ├── data/
│   ├── cross_validation.py
│   ├── data.py
│   ├── epochs.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── (submission files appear here once generated)
└── README.md
```

---

# **1. Decision Tree Models (ID3, Bagging, Random Forest)**

Located in: **`/decisionTree/`**

These models operate on **pandas DataFrames**.

## File Overview

### **`model.py`**
Implements:
- Majority baseline  
- Core **ID3 Decision Tree**  
- Entropy + information gain  
- Recursive tree building  
- Predicting with leaf fallback  
- `init_ID3()` factory for selecting simple, bagging, or random forest variants

### **`bagging_id3.py`**
Implements:
- Bootstrap sampling of rows  
- Training multiple ID3 trees  
- Majority-vote prediction

### **`random_forest_id3.py`**
Implements:
- Bootstrap sampling  
- Per-node random feature subsampling  
- ID3 training with subset of features  
- Majority vote across trees

### **`train.py`**
Runs a full training pipeline:
1. Load training data (`--train_path`)  
2. Impute missing values  
3. Instantiate selected model  
4. Train  
5. Load test set  
6. Predict labels  
7. Build **submission.csv**

### **`cross_validation.py`**
Performs K-fold CV on ID3, Bagging, or Random Forest using pre-generated folds.

### **`data.py`**
Loads CV folds stored in CSV format (simple version).  
Uses **mode** imputation.

### **`data_advanced.py`**
Loads full train + test set (for XGBoost).  
Uses **median** imputation.  
Designed for feature-matrix (NumPy) workflow.

### **`train_xgboost.py`**
Implements:
- Median imputation  
- XGBoost classifier  
- Submission generation

---

# ▶**How to Run the Decision Tree Models**

### **1. Run Cross-Validation (ID3/Bagging/Random Forest)**

```
python cross_validation.py     --cv_path data/cv     --model simple     --num_trees 25
```

### **2. Train Final Model and Generate Submission**

```
python train.py     --model bagging     --train_path data/train_original.csv     --eval_path data/test_original.csv     --ids_path data/eval.ids     --depth_limit 8     --num_trees 25
```

Output written to:

```
submission.csv
```

### **3. XGBoost Baseline**

```
python train_xgboost.py
```

Outputs:

```
submission_xgboost.csv
```

---

# **2. Perceptron Models**

Located in: **`/perceptron/`**

These models operate on **NumPy arrays** and include full perceptron variants taught in class.

## File Overview

### **`model.py`**
Implements:
- Majority baseline  
- Simple Perceptron  
- Decayed-LR Perceptron  
- Margin Perceptron  
- Averaged Perceptron  

### **`data.py`**
Loads:
- train_original.csv  
- test_original.csv  
- eval.ids  
- CV folds  

Uses median imputation.

### **`evaluate.py`**
Contains:
- `accuracy()`  
- `f1_macro()`  

### **`epochs.py`**
Tool for determining optimal epoch count using validation split.

### **`cross_validation.py`**
Grid-search CV over:
- Learning rate  
- Margin μ  
- Epoch count  
- Model variant  

### **`train.py`**
Final training and submission generator.

---

# **How to Run Perceptron Models**

### **1. Run Cross-Validation Grid Search**

```
python cross_validation.py     --model simple     --lr_values 1 0.1     --mu_values 0 1     --epochs 10
```

### **2. Find Optimal Epoch Count**

```
python epochs.py     --model simple     --lr 1     --mu 0     --epochs 20
```

### **3. Train Final Perceptron + Generate Submission**

```
python train.py     --model averaged     --lr 1     --mu 0     --epochs 10
```

---

# **3. Generate Folds (Optional)**

```
python generate_folds.py
```

This creates stratified CV folds under `data/cv/`.

---

# **4. Notes on Imputation & Leakage**

✔ Median imputation  
✔ Fit imputers on training folds only  
✔ No leakage into validation/test  

---

# **5. Required Models (Assignment)**

This project includes ≥5 models:

1. ID3  
2. Bagging (ID3)  
3. Random Forest (ID3)  
4. Perceptron variants  
5. XGBoost  

Meets all assignment requirements.

---

# **6. Submission Files**

Both pipelines output:

```
example_id,label
```

IDs match ordering of `eval.ids`.

---

