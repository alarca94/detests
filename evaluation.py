import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from utils import read_results


class Evaluator:
    def __init__(self, team_name, results_path='./'):
        self.team_name = team_name
        self.results_path = results_path
        self.A = 0.55
        self.B = 1.5
        self.alpha1 = 2
        self.alpha2 = 2
        self.beta = 3

    def propensity_f(self, y, y_pred):
        """
        This Propensity F measure assumes all classes are binary and encoded with 0s and 1s
        Note 1: True and False are considered separate classes
        Note 2: Root is added as a smoothing term
        Note 3: this is not a hierarchical metric
        """
        # Add empty set (all ones for smoothing) and false class (1st column == 0)
        empty_set = np.ones((y.shape[0], 1))
        y = np.hstack((empty_set, np.abs(y[:, :1] - 1), y))
        y_pred = np.hstack((empty_set, np.abs(y_pred[:, :1] - 1), y_pred))

        Nc = y.sum(axis=0, keepdims=True)  # Only ones are considered
        gs_c_ones = y.astype(bool)
        pred_c_ones = y_pred.astype(bool)
        hit_mask = np.logical_and(gs_c_ones, pred_c_ones)

        C = (np.log2(y.shape[0]) - 1) * ((self.B + 1) ** self.A)
        pc = 1 / (1 + C * np.exp(-self.A * np.log2(Nc + self.B)))

        inv_pc = 1 / pc
        inv_pc = np.broadcast_to(inv_pc, y.shape)
        numerator = np.multiply(inv_pc, hit_mask).sum(1)
        prop_p = numerator / np.multiply(inv_pc, pred_c_ones).sum(1)
        prop_r = numerator / np.multiply(inv_pc, gs_c_ones).sum(1)
        prop_f = 2 * prop_p * prop_r / (prop_p + prop_r)

        return prop_f.mean()

    def hierarchical_f(self, y, y_pred):
        # Add empty set (all ones for smoothing) and false class (1st column == 0)
        empty_set = np.ones((y.shape[0], 1))
        y = np.hstack((empty_set, np.abs(y[:, :1] - 1), y))
        y_pred = np.hstack((empty_set, np.abs(y_pred[:, :1] - 1), y_pred))

        hit_mask = np.logical_and(y.astype(bool), y_pred.astype(bool))
        n_hits = hit_mask.sum(1)

        p_f = n_hits / y_pred.sum(1)
        r_f = n_hits / y.sum(1)
        f_f = 2 * p_f * r_f / (p_f + r_f)

        return f_f.mean()

    def icm(self, y, y_pred):
        def ic(sets, ic_nodes):
            ic_sets = np.multiply(sets.astype(bool), ic_nodes)
            # In our scenario only True has descendants (the categories), so we need to sum the IC of all categories
            ic_descendents = ic_sets[:, 3:].sum(1)
            # The IC of the ascendant (True) of each pair needs to be subtracted |C| - 1 times
            n_categories = sets[:, 3:].sum(1)
            total_ic_sets = ic_descendents - (n_categories - 1) * ic_sets[:, 2]
            # If there are no categories, we will have only have False in the set. Need to add IC(False)
            total_ic_sets += ic_sets[:, 1]
            # Remember that the IC(ROOT) = 0, and the lso(False, any_category) = ROOT so there is no need to subtract it
            return total_ic_sets

        # Add empty set (all ones for smoothing) and false class (1st column == 0)
        empty_set = np.ones((y.shape[0], 1))
        y = np.hstack((empty_set, np.abs(y[:, :1] - 1), y))
        y_pred = np.hstack((empty_set, np.abs(y_pred[:, :1] - 1), y_pred))

        # [ROOT, FALSE, TRUE, C1, C2, ..., C10]
        ic_c = np.broadcast_to(-np.log2(y.sum(0) / y.shape[0]), y.shape)

        ic_y = ic(y, ic_c)
        ic_y_pred = ic(y_pred, ic_c)
        ic_y_union_y_pred = ic(np.logical_or(y, y_pred), ic_c)

        icm = self.alpha1 * ic_y + self.alpha2 * ic_y_pred - self.beta * ic_y_union_y_pred
        return icm.mean()

    def read_sort_check(self, pred_filename, gs_filename, task_number):
        pred_res = read_results(task_number=task_number, filename=pred_filename, results_path=self.results_path)
        gs_res = read_results(task_number=task_number, filename=gs_filename, results_path=self.results_path)

        pred_res.sort_values(by='sentence_id', inplace=True)
        gs_res.sort_values(by='sentence_id', inplace=True)

        assert gs_res['sentence_id'].shape[0] == pred_res['sentence_id'].shape[0]
        assert all(gs_res['sentence_id'].isin(pred_res['sentence_id']))
        assert all(pred_res['sentence_id'].isin(gs_res['sentence_id']))
        assert gs_res['sentence_id'].nunique() == gs_res['sentence_id'].shape[0]
        assert pred_res['sentence_id'].nunique() == pred_res['sentence_id'].shape[0]
        assert all(gs_res['sentence_id'].eq(pred_res['sentence_id'].unique()))
        assert pd.isna(gs_res.values).sum() + pd.isna(pred_res.values).sum() == 0
        assert all(np.isin(np.unique(gs_res.iloc[:, 1:].values), np.array([0, 1])))
        assert all(np.isin(np.unique(pred_res.iloc[:, 1:].values), np.array([0, 1])))
        if task_number == 2:
            assert all(pred_res[pred_res['stereotype'] == 0].iloc[:, 2:].values.flatten() == 0)
            assert all(gs_res[gs_res['stereotype'] == 0].iloc[:, 2:].values.flatten() == 0)

        return pred_res, gs_res

    def evaluate_task1(self, attempt, gs_filename):
        filename = f'{self.team_name}_task1_{attempt}.csv'
        pred_res, gs_res = self.read_sort_check(filename, gs_filename, task_number=1)

        return {
            'f_score': f1_score(gs_res['stereotype'].values, pred_res['stereotype'].values, average='binary'),
            'precision': precision_score(gs_res['stereotype'].values, pred_res['stereotype'].values, average='binary'),
            'recall': recall_score(gs_res['stereotype'].values, pred_res['stereotype'].values, average='binary')
        }

    def evaluate_task2(self, attempt, gs_filename):
        filename = f'{self.team_name}_task2_{attempt}.csv'
        pred_res, gs_res = self.read_sort_check(filename, gs_filename, task_number=2)

        target_cols = ['stereotype', 'xenophobia', 'suffering', 'economic', 'migration',
                       'culture', 'benefits', 'health', 'security', 'dehumanisation', 'others']

        y = gs_res[target_cols].values
        y_pred = pred_res[target_cols].values

        return {
            'hierarchical_f': self.hierarchical_f(y, y_pred),
            'propensity_f': self.propensity_f(y, y_pred),
            'icm': self.icm(y, y_pred)
        }
