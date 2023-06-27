import os

from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')

n_splits = 5
n_reps = 2
# a few constants
RANDOM_STATE = 42
N_ESTIMATORS = 50
k_folds = 5
n_repeats = 2
base_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)


if __name__ == '__main__':
    print(ROOT_DIR)