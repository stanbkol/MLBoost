import csv
import io
import numpy as np
import re
import os
import pandas as pd

from definitions import OUTPUT_DIR, ROOT_DIR


def wilcoxon_to_csv(measures, clf_names, fp="wilcoxon.csv"):
    measures = measures.round(decimals=3)
    df = pd.DataFrame(measures, columns=clf_names)
    df.insert(loc=0, column='Classifiers', value=np.array(clf_names).T)
    if fp is not None:
        df.to_csv(fp, index=False)


def scores_to_csv(mean_scores, mean_ranks, clf_names, datasets, fp="score_results.csv"):
    mean_scores = mean_scores.round(decimals=3)
    mean_ranks = mean_ranks.round(decimals=3)
    dts = np.array(datasets).T
    stat_df = pd.DataFrame(mean_scores, columns=clf_names)
    stat_df.insert(loc=0, column='Datasets', value=dts)
    ranks = mean_ranks.tolist()
    ranks = ["Mean Ranks"] + ranks
    stat_df.loc[len(stat_df)] = ranks
    # stat_df = stat_df.append(ranks, ignore_index=True)
    # stat_df.loc[len(stat_df.index)] = mean_ranks.tolist().insert(0, "Mean Rank")
    print(f"score df: \n{stat_df}")

    stat_df.to_csv(fp, index=False)


def array_to_csv(array: np.array, headers=None, fp="results.csv"):
    table = pd.DataFrame(array)
    print(f"table: \n{table}")

    if headers is not None:
        table.to_csv(fp, index=False, header=headers)
    else:
        table.to_csv(fp, index=False, header=False)


def create_score_table(classifiers, scores):
    clf_names = get_classifier_names(classifiers)
    pass


def get_classifier_names(classifiers: dict):
    return list(classifiers.keys())


def save_results(data: np.array, fn="results"):
    np.save(os.path.join(OUTPUT_DIR, fn), data)


def load_score_means(fp):
    scores = np.load(fp)
    print(f"scores shape: {scores.shape}")
    # classifiers x datasets x folds

    # shape = (datasets, classifiers)
    mean_scores = np.mean(scores, axis=2).T
    print(f"mean_scores: {mean_scores}")
    return mean_scores


def fetch_dataset_classes(file_path):
    with open(file_path) as file:
        while True:
            line = file.readline()
            if line.startswith("@"):
                if "attribute class" in line:
                    classes = line.split()[-1]
                    return re.sub(r"[{,}]", " ", classes).split()
            else:
                break
        print("no class attribute found")
        return None


def fetch_data(file_path, binary=True):
    if binary:
        return np.genfromtxt(file_path, comments='@', delimiter=',',
                             converters={(-1): lambda s: 0.0 if ("negative" in s.strip().decode('ascii')) else 1.0})
    # not finished
    if not binary:
        classes = fetch_dataset_classes(file_path)
        return np.genfromtxt(file_path, comments='@', delimiter=',',
                             converters={(-1): lambda s: classes.index(s.strip().decode('ascii')) if (
                                     s.strip().decode('ascii') in classes) else -1})


if __name__ == '__main__':
    file = io.StringIO(u"6,154,78,41,140,46.1,0.571,27,tested_negative")
    print(fetch_data(file))

    text = "{L,G,B}"
    res = re.sub(r"[{,}]", " ", text).split()
    print(res)
