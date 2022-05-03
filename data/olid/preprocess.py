import os.path

import pandas as pd

file = os.path.join(".", "olid-training-v1.0.tsv")
data = pd.read_csv(file, names=['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c'], sep='\t')
data_a = data.copy()
data_a = data_a.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})
sub_data_a = data_a[['text', 'labels']]
df_a = pd.DataFrame(sub_data_a)
df_a.to_csv('olid-data_sub_task_a.tsv', sep='\t', index=False)

data_b = data.copy()
data_b = data_b[['tweet', 'subtask_b']]
data_b = data_b.rename(columns={'subtask_b': 'labels', 'tweet': 'text'})
sub_data_b = data_b[['text', 'labels']]
df_b = pd.DataFrame(sub_data_b)
df_b.to_csv('olid-data_sub_task_b.tsv', sep='\t', index=False)

data_c = data.copy()
data_c = data_c[['tweet', 'subtask_c']]
data_c = data_c.rename(columns={'subtask_c': 'labels', 'tweet': 'text'})
sub_data_c = data_c[['text', 'labels']]
df_c = pd.DataFrame(sub_data_c)
df_c.to_csv('olid-data_sub_task_c.tsv', sep='\t', index=False)