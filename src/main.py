from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from classifiers import *
from datasets import *
from definitions import *
from src.statistical_tests import wilcoxon_test, friedman_test
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import numpy as np
from sklearn.base import clone
from collections import Counter
from tqdm import tqdm
from utils import *

#TODO: add more chart

metrics = {
    'g': "gmean",
    'f': "f1",
    'p': "precision",
    'r': "recall",
    'a': "accuracy"
}
cross_validator = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_reps, random_state=RANDOM_STATE)
g_mean_scores = np.zeros((len(classifiers), len(binary_datasets), n_reps * k_folds))
acc_scores = np.zeros((len(classifiers), len(binary_datasets), n_reps * k_folds))
f1_scores = np.zeros((len(classifiers), len(binary_datasets), n_reps * k_folds))
precision_scores = np.zeros((len(classifiers), len(binary_datasets), n_reps * k_folds))
recall_scores = np.zeros((len(classifiers), len(binary_datasets), n_reps * k_folds))


def generate_results():
    for data_id, dataset_name in tqdm(enumerate(binary_datasets), total=len(binary_datasets),
                                      desc=f"process dataset.."):
        print(f'DATASET {dataset_name.upper()}')
        dataset_path = os.path.join(DATA_DIR, f"{dataset_name}.dat")
        dataset = fetch_data(dataset_path)
        # print(f"dataset shape: {dataset.shape}")

        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fid, (train, test) in enumerate(cross_validator.split(X, y)):
            for clf_id, clf_name in enumerate(classifiers):
                clf = clone(classifiers[clf_name])
                # print(f"ytrian: {np.unique(y).size}")
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                acc_scores[clf_id, data_id, fid] = accuracy_score(y[test], y_pred)
                g_mean_scores[clf_id, data_id, fid] = geometric_mean_score(y[test], y_pred)
                precision_scores[clf_id, data_id, fid] = precision_score(y[test], y_pred)
                recall_scores[clf_id, data_id, fid] = recall_score(y[test], y_pred)
                f1_scores[clf_id, data_id, fid] = f1_score(y[test], y_pred)

    save_results(acc_scores, "acc_results")
    save_results(g_mean_scores, "gmean_results")
    save_results(f1_scores, "f1_results")
    save_results(recall_scores, "recall_results")
    save_results(precision_scores, "precision_results")


def metric_stats(score_means, clf_names, m=metrics['g']):
    mean_ranks, ranks = friedman_test(clf_names, score_means)
    statistics, p_scores = wilcoxon_test(classifiers, score_means)
    # wilcoxon_to_csv(p_scores, clf_names, fp=os.path.join(CSV_DIR, f"{m}_pscores.csv"))
    # wilcoxon_to_csv(statistics, clf_names, fp=os.path.join(CSV_DIR, f"{m}_stats.csv"))
    # scores_to_csv(score_means, mean_ranks, clf_names, binary_datasets, os.path.join(CSV_DIR, f'{m}_scores.csv'))


def statistical_analysis():
    clfs_names = get_classifier_names(classifiers)
    print(clfs_names)
    # results files
    acc_dir = os.path.join(OUTPUT_DIR, "acc_results.npy")
    g_means_dir = os.path.join(OUTPUT_DIR, "gmean_results.npy")
    f_measures_dir = os.path.join(OUTPUT_DIR, "f1_results.npy")
    precision_dir = os.path.join(OUTPUT_DIR, "precision_results.npy")
    recall_dir = os.path.join(OUTPUT_DIR, "recall_results.npy")

    # load results
    acc_means = load_score_means(acc_dir)
    f_means = load_score_means(f_measures_dir)
    precision_means = load_score_means(precision_dir)
    recall_means = load_score_means(recall_dir)
    g_means = load_score_means(g_means_dir)

    metric_stats(acc_means, clfs_names, m=metrics['a'])
    metric_stats(g_means, clfs_names, m=metrics['g'])
    metric_stats(f_means, clfs_names, m=metrics['f'])
    metric_stats(precision_means, clfs_names, m=metrics['p'])
    metric_stats(recall_means, clfs_names, m=metrics['r'])

    # print(f"mean_ranks: {mean_ranks}")
    # class_ranks = np.column_stack((clfs_names, np.round(mean_ranks, decimals=3)))
    # class_ranks = class_ranks[class_ranks[:, 1].argsort()]
    # headers = ["Classifier", "Mean Rank"]
    # print(f"class_ranks: {class_ranks}")
    # class_ranks_csv = os.path.join(CSV_DIR, 'class_mean_ranks.csv')
    # array_to_csv(class_ranks, headers=headers, fp=class_ranks_csv)


def run():
    generate_results()
    statistical_analysis()


if __name__ == '__main__':
    run()
