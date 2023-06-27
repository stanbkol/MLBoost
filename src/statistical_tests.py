import numpy as np
import pandas as pd
from scipy.stats import rankdata, ranksums, wilcoxon
from tabulate import tabulate
from scipy import stats
from src.utils import get_classifier_names

float_format = ".3f"


def friedman_test(clf_names, mean_scores):
    print("\nStart the FRIEDMAN TEST already!")
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print(f"\nRanks:\n {ranks}, {ranks.shape}")

    mean_ranks = np.mean(ranks, axis=0)  # by column
    # print(f"\nMean ranks:\n {mean_ranks}, {mean_ranks.shape}")
    print("Mean Ranks:")
    class_ranks = zip(clf_names, mean_ranks)
    class_ranks = sorted(class_ranks, key=lambda x: x[1])
    for name, rank in class_ranks:
        print(f"{name}: {round(rank, 3)}")
    return mean_ranks.round(decimals=4), ranks


def wilcoxon_test(clf_names, mean_scores: np.array):
    print("Start the WILCOXON TEST already!")
    num_classes = len(clf_names)
    statistics = np.zeros((num_classes, num_classes))
    p_values = np.zeros((num_classes, num_classes))
    # z_statistics = np.zeros((num_classes, num_classes))

    mean_scores = mean_scores.T
    # print(mean_scores.T)
    for i in range(len(clf_names)):
        for j in range(len(clf_names)):
            i_mean_scores = mean_scores[i]
            j_mean_scores = mean_scores[j]
            if i != j:
                res = wilcoxon(i_mean_scores, j_mean_scores)
                statistics[i, j], p_values[i, j] = res.statistic, res.pvalue
                # statistics[i, j], p_values[i, j], z_statistics[i, j] = stats.wilcoxon(i_mean_scores, j_mean_scores)

    return statistics, p_values


def stat_pval_tables(classifiers, statistics, p_values):
    headers = get_classifier_names(classifiers)
    measures = statistics.round(decimals=4)
    statistic_table = pd.DataFrame(statistics, columns=headers)
    statistic_table.insert(loc=0, column='Classifiers', value=np.array(headers).T)

    p_values_table = pd.DataFrame(p_values, columns=headers)
    p_values_table.insert(loc=0, column='Classifiers', value=np.array(headers).T)

    return statistic_table, p_values_table


