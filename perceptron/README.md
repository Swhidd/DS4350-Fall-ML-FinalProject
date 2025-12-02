# Assignment 4

Functions to implement:

```
model.py
    MajorityBaseline.train()
    MajorityBaseline.predict()
    Perceptron.train()
    Perceptron.predict()
    AveragedPerceptron.train()
    AveragedPerceptron.predict()
    AggressivePerceptron.train()
    AggressivePerceptron.predict()

cross_validation.py
    cross_validation()

epochs.py
    train_epochs()

Note: Do not change anything in `data.py` or `evaluate.py` or `train.py`.
```

### Setup and Installation

Before you begin, you need to set up a virtual environment to manage the project's packages.

1.  **Create a Virtual Environment:**
    From your main project directory (the one containing the `assignment4` folder), run the following command to create a virtual environment named `venv`:
    ```sh
    python3 -m venv venv
    ```

2.  **Activate the Virtual Environment:**
    You must activate the environment every time you work on the project.

    **On macOS / Linux:**
    ```sh
    source venv/bin/activate
    ```
    **On Windows:**
    ```sh
    .\venv\Scripts\activate
    ```    You will see `(venv)` at the beginning of your terminal prompt when it's active

3.  **Navigate into the Assignment Folder:**
    All subsequent commands must be run from inside the `assignment4` directory.
    ```sh
    cd assignment4
    ```

4.  **Install Required Packages:**
    Install the necessary packages using `pip` and the provided `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```

You are now ready to run the scripts.

## Models

### Majority Baseline

Once you've implemented `MajorityBaseline`, you can train and evaluate your model with:
```sh
python train.py -m majority_baseline
# To get the development accuracy:
python -c "import numpy as np; import pandas as pd; from data import load_data; from model import MajorityBaseline; from evaluate import accuracy; D=load_data(); tr,va=D['train'],D['val']; Xtr=tr.drop(columns=['label']).to_numpy(); ytr=tr['label'].tolist(); Xva=va.drop(columns=['label']).to_numpy(); yva=va['label'].tolist(); clf=MajorityBaseline(); clf.train(Xtr,ytr); preds=clf.predict(Xva); tolist=lambda a:(a if isinstance(a,list) else (a.ravel().tolist() if hasattr(a,'ravel') else (a.tolist() if hasattr(a,'tolist') else list(a)))); print('Majority baseline DEV accuracy =', accuracy(tolist(yva), tolist(preds)))"
```
Make sure your code works for `MajorityBaseline` before moving on to `Perceptron`. 

### Perceptron (Simple, Decay, Margin)

The `Perceptron` class will handle the Simple Perceptron, the Decaying Learning Rate Perceptron, and the Margin Perceptron. You can initialize these in the `train.py` script with the `-m` flag set to `simple`, `decay`, or `margin`. The `--lr` and `--mu` flags set the learning rate and mu hyperparameters, respectively. For the `simple` and `decay` models, you can either ignore the `self.mu` class variable, or use it in your linear threshold with `mu=0`. For the decay model, the `lr` parameter is the same as the lr0 hyperparameter. The `num_features` parameter will let you determine the number of weights your model will have. It's up to you whether you have an explicit bias term or whether you fold it into your weights.

Once you've implemented `Perceptron`, you can train and evaluate your model like this(The lr/mu values below are examples for testing your model.):
```sh
# train/eval a simple perceptron with lr=1 for 10 epochs
python train.py -m simple --lr 1 --epochs 10

# train/eval a decay perceptron with lr=0.1 for 10 epochs
python train.py -m decay --lr 0.1 --epochs 10

# train/eval a margin perceptron with lr=1 and mu=0.5 for 10 epochs
python train.py -m margin --lr 1 --mu 0.5 --epochs 10
```

### Averaged Perceptron

The `AveragedPerceptron` class will handle the Averaged Perceptron. Once you've implemented `AveragedPerceptron`, you can train and evaluate your model like this:
```sh
# train/eval an averaged perceptron with lr=1 for 10 epochs
python train.py -m averaged --lr 1 --epochs 10
```

### Aggressive Perceptron (Bonus)

The `AggressivePerceptron` class will handle the Aggressive Perceptron. Once you've implemented `AggressivePerceptron`, you can train and evaluate your model like this:
```sh
# train/eval an aggressive perceptron with mu=1 for 10 epochs
python train.py -m aggressive --mu 1 --epochs 10
```

## Cross Validation

After implementing the code in `cross_validation.py`, you can perform a grid search to find the best hyperparameters. The script will test all combinations of the learning rates and margins you provide.
```sh
# for models that don't use a hyperparameter, you can omit it
# run cross validation for variants with lr values [1, 0.1, 0.01]
# and mu values [1, 0.5, 0.1, 0.01] for 10 epochs
python cross_validation.py -m simple --lr_values 1 0.1 0.01 --epochs 10
python cross_validation.py -m decay --lr_values 1 0.1 0.01 --epochs 10
python cross_validation.py -m margin --lr_values 1 0.1 0.01 --mu_values 1 0.5 0.1 0.01 --epochs 10
python cross_validation.py -m averaged --lr_values 1 0.1 0.01 --epochs 10
# for bonus
python cross_validation.py -m aggressive --mu_values 1 0.5 0.1 0.01 --epochs 10
```

## Epoch Training

In this part, you'll use the train and validation datasets to see how long to train your perceptron for. You'll want to train the same model for one epoch at a time, evaluate against the validation dataset, and see which epoch yields the best validation accuracy. Beware of "off-by-one" errors. Once you've implemented the necessary code in `epochs.py`, you can run epoch training with:
```sh
# run epoch training for models for 20 epochs with lr and mu you get in the Cross Validation 
# Best lr was 0.01 w/ accuracy 0.985
python epochs.py -m simple --lr 0.01 --epochs 20
# Best lr was 0.01 w/ accuracy 0.982
python epochs.py -m decay --lr 0.01 --epochs 20
# Best lr was 1.0 and mu 0.01 w/ accuracy 0.985
python epochs.py -m margin --lr 1.0 --mu 0.01 --epochs 20
# Best lr was 0.1 w/ accuracy 0.989
python epochs.py -m averaged --lr 0.1 --epochs 20
# For bonus:
python epochs.py -m aggressive --mu <BEST_MU_HERE> --epochs 20 
```

## Final Evaluation

After running cross-validation to find the best hyperparameters and epoch training to find the optimal number of epochs, run the training script one last time with the best settings to get the final test accuracies for your report.
```sh
# final run
# Best lr was 0.01 for 3 epochs w/ accuracy 0.985   Final train/test: 0.981/0.976
python train.py -m simple --lr 0.01 --epochs 3
# Best lr was 0.01 for 2 epochs w/ accuracy 0.990   Final train/test: 0.981/0.976
python train.py -m decay --lr 0.01 --epochs 2
# Best lr was 1.0 and mu 0.01 for 6 epochs w/ accuracy 0.990    Final train/test: 0.980/0.985
python train.py -m margin --lr 1.0 --mu 0.01 --epochs 6
# Best lr was 0.1 for 8 epochs w/ accuracy 0.966    Final train/test: 0.989/0.981
python train.py -m averaged --lr 0.1 --epochs 8
# For bonus:
python train.py -m aggressive --mu <BEST_MU_HERE> --epochs <BEST_EPOCHS_HERE>
```
