import os.path

import pandas as pd

file = os.path.join("data", "other", "olid-training-v1.0.tsv")
data = pd.read_csv(file, usecols=[0, 1, 2], sep='\t', encoding='utf-8')
sub_data_a = data[['tweet', 'subtask_a']]
sub_data_a = sub_data_a.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})
sub_a_file_path = os.path.join("data", "other", "olid-data_sub_task_a.tsv")
sub_data_a.to_csv(sub_a_file_path, sep='\t', encoding='utf-8',index=False)


file = os.path.join("data", "other", "hasoc2019_hi_test_gold_2919.tsv")
data = pd.read_csv(file, usecols=[0, 1, 2], sep='\t', encoding='utf-8')
sub_data_a = data[['tweet', 'subtask_a']]
sub_data_a = sub_data_a.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})
sub_a_file_path = os.path.join("data", "hin-data_sub_task_a.tsv")
sub_data_a.to_csv(sub_a_file_path, sep='\t', encoding='utf-8',index=False)

# data = pd.read_csv(file, usecols=[0, 1, 3], sep='\t')
# sub_data_b = data[['tweet', 'subtask_b']]
# sub_data_b = sub_data_b.rename(columns={'subtask_b': 'labels', 'tweet': 'text'})
# sub_data_b.to_csv('other-data_sub_task_b.tsv', sep='\t', index=False)
#
# data = pd.read_csv(file, usecols=[0, 1, 4], sep='\t')
# sub_data_b = data[['tweet', 'subtask_c']]
# sub_data_b = sub_data_b.rename(columns={'subtask_c': 'labels', 'tweet': 'text'})
# sub_data_b.to_csv('other-data_sub_task_c.tsv', sep='\t', index=False)
