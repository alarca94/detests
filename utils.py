import os
import pandas as pd


def read_results(task_number, filename, results_path):
    if task_number == 1:
        columns = ['sentence_id', 'stereotype']
    elif task_number == 2:
        columns = ['sentence_id', 'stereotype', 'xenophobia', 'suffering', 'economic', 'migration',
                   'culture', 'benefits', 'health', 'security', 'dehumanisation', 'others']
    else:
        raise ValueError('Task number must be either 1 or 2')

    results = pd.read_csv(os.path.join(results_path, filename))
    results['sentence_id'] = results['comment_id'].astype(str) + '_' + results['sentence_pos'].astype(str)

    return results[columns]


def save_submission(results_df, team_name, task_number, results_path='./', attempt=0):
    if task_number == 1:
        columns = ['comment_id', 'sentence_pos', 'stereotype']
    elif task_number == 2:
        columns = ['comment_id', 'sentence_pos', 'stereotype', 'xenophobia', 'suffering', 'economic', 'migration',
                   'culture', 'benefits', 'health', 'security', 'dehumanisation', 'others']
    else:
        raise ValueError(f'The task number must be 1 or 2')

    results_df = results_df[columns]
    assert pd.isna(results_df).values.sum() == 0
    results_df.to_csv(os.path.join(results_path, f'{team_name}_task{task_number}_{attempt}.csv'), index=False)
