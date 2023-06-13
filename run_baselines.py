import os
import random
import spacy
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from evaluation import Evaluator
from utils import save_submission


def run_task1(model='all'):
    np.random.seed(41)

    DATA_PATH = './data'
    train_file = 'train.csv'
    test_file = 'test.csv'

    trn = pd.read_csv(os.path.join(DATA_PATH, train_file))
    tst = pd.read_csv(os.path.join(DATA_PATH, test_file))

    target_col = 'stereotype'
    feat_cols = 'sentence'
    results = tst[['comment_id', 'sentence_pos']]

    # Run TF-IDF + SVM Baseline
    if model in ['tfidf', 'all']:
        team_name = 'TFIDF+SVC'
        pipe = Pipeline([('tfidf', TfidfVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1, 3),
                                                   max_features=10000)),
                         ('svc', SVC(kernel='linear'))])
        pipe.fit(trn[feat_cols].values, trn[target_col].values)
        results[target_col] = pipe.predict(tst[feat_cols]).tolist()

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=1,
                        results_path='./results/',
                        attempt=0)

    # Weighted Random Classifier
    if model in ['random', 'all']:
        team_name = 'RandomClassifier'
        v, p = np.unique(trn[target_col].values, return_counts=True)
        p = p / trn.shape[0]
        results[target_col] = np.random.choice(v, size=(tst.shape[0], ), p=p)

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=1,
                        results_path='./results/',
                        attempt=0)

    # All Ones
    if model in ['ones', 'all']:
        team_name = 'AllOnes'
        results[target_col] = 1
        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=1,
                        results_path='./results/',
                        attempt=0)

    # Word2Vec
    if model in ['w2v', 'all']:
        team_name = "W2V+SVC"
        nlp = spacy.load("es_core_news_md")
        def get_sent_emb(sent):
            toks = nlp(sent)
            return np.stack([t.vector for t in toks]).mean(0)
        X = np.stack(trn[feat_cols].apply(get_sent_emb).values)
        cls = SVC(kernel='linear')
        cls.fit(X, trn[target_col].values)
        results[target_col] = cls.predict(np.stack(tst[feat_cols].apply(get_sent_emb).values))

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=1,
                        results_path='./results/',
                        attempt=0)

    # Gold Standard
    if model in ['gs', 'all']:
        team_name = "GoldStandard"
        tst_full = pd.read_csv(os.path.join(DATA_PATH, 'test_full.csv'))
        results[target_col] = tst_full[target_col]

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=1,
                        results_path='./results/',
                        attempt=0)


def run_task2(model='all'):
    np.random.seed(41)

    DATA_PATH = './data'
    train_file = 'train.csv'
    test_file = 'test.csv'

    trn = pd.read_csv(os.path.join(DATA_PATH, train_file))
    tst = pd.read_csv(os.path.join(DATA_PATH, test_file))

    t1_target_col = 'stereotype'
    t2_target_cols = ['xenophobia', 'suffering', 'economic', 'migration', 'culture',
                      'benefits', 'health', 'security', 'dehumanisation', 'others']
    feat_cols = 'sentence'
    results = tst[['comment_id', 'sentence_pos']]

    # Run TF-IDF + SVM Baseline
    if model in ['tfidf', 'all']:
        team_name = 'TFIDF+SVC'
        pipe = Pipeline([('tfidf', TfidfVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1, 3),
                                                   max_features=10000)),
                         ('svc', SVC(kernel='linear'))])
        pipe.fit(trn[feat_cols].values, trn[t1_target_col].values)
        results[t1_target_col] = pipe.predict(tst[feat_cols]).tolist()

        t2_trn = trn[trn[t1_target_col] == 1]
        tst_mask = results[t1_target_col] > 0.5
        t2_tst = tst[tst_mask]
        for c in t2_target_cols:
            results[c] = 0
            pipe = Pipeline([('tfidf', TfidfVectorizer(strip_accents='unicode', lowercase=True, ngram_range=(1, 3),
                                                       max_features=10000)),
                             ('svc', SVC(kernel='linear'))])
            pipe.fit(t2_trn[feat_cols].values, t2_trn[c].values)
            results.loc[tst_mask, c] = pipe.predict(t2_tst[feat_cols]).tolist()

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)

    # Weighted Random Classifier
    if model in ['random', 'all']:
        team_name = 'RandomClassifier'
        v, p = np.unique(trn[t1_target_col].values, return_counts=True)
        p = p / trn.shape[0]
        results[t1_target_col] = np.random.choice(v, size=(tst.shape[0],), p=p)

        t2_trn = trn[trn[t1_target_col] == 1]
        tst_mask = results[t1_target_col] > 0.5
        t2_tst = tst[tst_mask]
        for c in t2_target_cols:
            results[c] = 0
            v, p = np.unique(t2_trn[c].values, return_counts=True)
            p = p / t2_trn.shape[0]
            results.loc[tst_mask, c] = np.random.choice(v, size=(t2_tst.shape[0],), p=p)

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)

    # All Ones
    if model in ['ones', 'all']:
        team_name = 'AllOnes'
        results[t1_target_col] = 1
        for c in t2_target_cols:
            results[c] = 1

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)

    # All Zeros
    if model in ['zeros', 'all']:
        team_name = 'AllZeros'
        results[t1_target_col] = 0
        for c in t2_target_cols:
            results[c] = 0

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)

    # W2V
    if model in ['w2v', 'all']:
        team_name = "W2V+SVC"
        nlp = spacy.load("es_core_news_md")

        def get_sent_emb(sent):
            toks = nlp(sent)
            return np.stack([t.vector for t in toks]).mean(0)

        X_trn = np.stack(trn[feat_cols].apply(get_sent_emb).values)
        X_tst = np.stack(tst[feat_cols].apply(get_sent_emb).values)
        cls = SVC(kernel='linear')
        cls.fit(X_trn, trn[t1_target_col].values)
        results[t1_target_col] = cls.predict(X_tst)

        trn_mask = trn[t1_target_col] == 1
        t2_trn = trn[trn_mask]
        tst_mask = results[t1_target_col] > 0.5
        for c in t2_target_cols:
            results[c] = 0
            cls = SVC(kernel='linear')
            cls.fit(X_trn[trn_mask.values], t2_trn[c].values)
            results.loc[tst_mask, c] = cls.predict(X_tst[tst_mask]).tolist()

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)

    # Gold Standard
    if model in ['gs', 'all']:
        team_name = "GoldStandard"
        tst_full = pd.read_csv(os.path.join(DATA_PATH, 'test_full.csv'))
        results[t1_target_col] = tst_full[t1_target_col]
        for c in t2_target_cols:
            results[c] = tst_full[c]

        save_submission(results_df=results,
                        team_name=team_name,
                        task_number=2,
                        results_path='./results/',
                        attempt=0)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None

    model = 'all'
    run_task1(model)
    run_task2(model)
