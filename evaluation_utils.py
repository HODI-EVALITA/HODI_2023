import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys
from scipy.stats import sem
import numpy as np
from ast import literal_eval


def check_submission_consistency(gold_df, submission_df, levels, task):
    # Check length files
    if (len(gold_df) != len(submission_df)):
        raise Exception('Prediction and gold data have different number of lines.')

    if task.lower() == "a":
        # Check predicted classes
        for c in levels:
            gt_class = list(gold_df[c].value_counts().keys())
            if not (submission_df[c].isin(gt_class).all()):
                raise Exception("Wrong value in " + c + " prediction column of data.")


def check_merge_length(ground_truth, predicted):
    # Check length files
    if (len(ground_truth) != len(predicted)):
        raise Exception('Prediction and gold data have different number of lines or different IDs.')


def get_metric_subtask_a(data):
    levels = ["homotransphobic"]
    acc_levels = dict.fromkeys(levels)
    p_levels = dict.fromkeys(levels)
    r_levels = dict.fromkeys(levels)
    f1_levels = dict.fromkeys(levels)
    for l in levels:
        acc_levels[l] = accuracy_score(data[l], data[l + "_pred"])
        p_levels[l], r_levels[l], f1_levels[l], _ = precision_recall_fscore_support(data[l], data[l + "_pred"],
                                                                                    average="macro")
    macro_f1 = np.mean(list(f1_levels.values()))
    return macro_f1, f1_levels


def evaluate_task_a_singlefile(data_predicted_file, data_gold, task):
    check_submission_consistency(data_gold, data_predicted_file, ['homotransphobic'], task)

    ## Compute macro F1
    results_raw = pd.merge(data_gold, data_predicted_file, on="id", suffixes=("", "_pred"))
    check_merge_length(results_raw, data_gold)
    macro_f1, f1_levels = get_metric_subtask_a(results_raw)
    print("taskA_fscore_homotransphobic: {0}\n".format(f1_levels["homotransphobic"]))

    return macro_f1


def evaluate_task_b_singlefile(data_predicted_file, data_gold, task):
    check_submission_consistency(data_gold, data_predicted_file, ['rationales'], task)

    ## Compute macro F1
    results_raw = pd.merge(data_gold, data_predicted_file, on="id", suffixes=("", "_pred"))
    check_merge_length(results_raw, data_gold)
    xx = get_metric_subtask_b(results_raw)
    print("taskb_agreement_rationales: {0}\n".format(xx))

    return xx


def f1(predictions, gold):
    """
	F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
	>>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
	:param predictions: a list of predicted offsets
	:param gold: a list of offsets serving as the ground truth
	:return: a score between 0 and 1
	"""
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom) / float(denom)


def get_metric_subtask_b(data):
    """
	Based on https://github.com/felipebravom/EmoInt/blob/master/codalab/scoring_program/evaluation.py
	"""
    scores = []
    for index, row in data.iterrows():
        gold_spans = row['rationales']
        pred_spans = row['rationales_pred']
        scores.append(f1(pred_spans, gold_spans))

    return np.mean(scores)
