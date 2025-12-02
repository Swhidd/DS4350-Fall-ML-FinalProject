# Assignment 2

Functions to implement:

```
model.py
    MajorityBaseline.train()
    MajorityBaseline.predict()
    DecisionTree.train()
    DecisionTree.predict()

train.py
    train()
    evaluate()
    calculate_accuracy()

cross_validation.py
    cross_validation()
```
Don't change anything in data.py

Setup and Installation-
Before you begin, you need to set up a virtual environment to manage the project's packages.

1.  Create a Virtual Environment:
    From the project's root directory, run the following command to create a virtual environment named `venv`:
    ```sh
    python3 -m venv venv
    ```

2.  Activate the Virtual Environment:
    You must activate the environment every time you work on the project.

    On macOS / Linux:
        ```sh
        source venv/bin/activate
        ```
    On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    You will see `(venv)` at the beginning of your terminal prompt when it's active.

3.  Install Required Packages
    Navigate into the `assignment2` folder and install the necessary packages using `pip`:
    ```sh
    pip install -r assignment2/requirements.txt
    ```

You are now ready to run the scripts.


Once you've implemented `MajorityBaseline` and the functions in `train.py`, you can train and evaluate your model with:
```sh
python assignment2/train.py -t data/train.csv -e data/test.csv -m "majority baseline"
```

Make sure your code works for `MajorityBaseline` before moving on to `DecisionTree`. 

Next, once you've completed `DecisionTree`, you can train and evaluate your model with:
```sh
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree"              # runs with no depth limiting
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree" -d 2         # run with a specific depth limit (e.g., 2)
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree" -i collision # runs with "collision entropy" as the ig_criterion instead of "entropy"
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree" -d 2 -i collision # runs with depth_limit=2 and ig_criterion="collision"
```

Once you've implemented the necessary code in `cross_validation.py`, you can run cross validation and can find the best depth for your decision tree. The script will test depths from 1 to 12 as required by the assignment.
```sh
python assignment2/cross_validation.py -c data/cv/ -i entropy # runs CV with entropy which is the default criterion
python assignment2/cross_validation.py -c data/cv/ -i collision # runs CV with collision entropy which is for the bonus
```

After running cross validation, you will get an optimal depth. Use that depth to run the training script one last time to get the final test accuracies for your report.
```sh
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree" -d <Put your BEST_DEPTH_HERE> # for the main task that is the entropy
python assignment2/train.py -t data/train.csv -e data/test.csv -m "decision tree" -d <Put your BEST_DEPTH_HERE> -i collision # for the bonus task that is the collision entropy
```
