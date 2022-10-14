import argparse
import os

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from offensive_nn.config.sold_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE

from scipy.special import softmax
import numpy as np

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--lang', required=False, help='language', default="en")  # en or sin
parser.add_argument('--algorithm', required=False, help='algorithm', default="cnn2D")  # lstm or cnn2D
parser.add_argument('--train', required=False, help='train file', default='data/olid/olid-training-v1.0.tsv')
parser.add_argument('--test', required=False, help='test file')
arguments = parser.parse_args()

trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

if arguments.lang == "en":
    trn_data, tst_data = train_test_split(trn_data, test_size=0.1)

elif arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'content': 'text', 'Class': 'labels'})

elif arguments.lang == "hin":
    trn_data = trn_data.rename(columns={'task_1': 'labels'})
    tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

train_set = trn_data[['text', 'labels']]
train_set['labels'] = encode(train_set['labels'])
test_set = tst_data[['text']]

test_sentences = test_set['text'].tolist()

test_preds = np.zeros((len(test_set), args["n_fold"]))

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = arguments.cuda_device

for i in range(args["n_fold"]):
    train_set, validation_set = train_test_split(train_set, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(MODEL_NAME,model_type_or_path=arguments.algorithm,
                             train_df=train_set,
                             args=args, eval_df=validation_set)
    model.train_model()
    print("Finished Training")
    model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
    predictions, raw_outputs = model.predict(test_sentences)
    probs = softmax(raw_outputs, axis=1)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

test_set['predictions'] = final_predictions

# select majority class of each instance (row)
# prediction_large_csv = test_set['predictions']
prediction_large_csv = test_set
prediction_large_csv.to_csv('best_model_prediction_large.csv')
confidence_df = pd.DataFrame(probs)
test_set['preds'] = predictions
predictions_df = pd.merge(test_set, test_set[['preds']], how='left', left_index=True, right_index=True)
predictions_df.to_csv('prediction.csv')
confidence_df.to_csv('confidence_result1.csv', index=False)
test_set['predictions'] = predictions
df1 = pd.read_csv('prediction.csv')
column_names = ['1', '2']
df = pd.read_csv('confidence_result1.csv', names=column_names, header=None)
frames = [df, df1]
result = pd.concat([df1, df], axis=1)
result.to_csv('one_prediction.csv')
# print((result['preds_y']).value_counts())

new = []
new1 = []
new2 = []

m1 = np.mean(df['1'])
m2 = np.mean(df['2'])

print(m1,m2)

l1 = 0.01
l2 = np.std(df['2'])

# get all the offensive and not offensive posts from the dataset

df_group_posts = result.groupby('preds_y')
offensive_posts = df_group_posts.get_group(0.0)
if offensive_posts is not None:
    for ix in offensive_posts.index:
        off_prob = offensive_posts.loc[ix]['1']
        if ((m1 + l1 > off_prob) and (m1 - l1 < off_prob)):
            new.append(offensive_posts.loc[ix]['1'])
else:
    new.append(None)

offensive_not_posts = df_group_posts.get_group(1.0)
if offensive_not_posts is not None:
    for ix in offensive_not_posts.index:
        not_off_prob = offensive_not_posts.loc[ix]['2']
        if ((m1 + l1 > not_off_prob) and (m1 - l1 < not_off_prob)):
            new2.append(offensive_not_posts.loc[ix]['2'])
else:
    new2.append(None)

df_new = result.iloc[np.where(result['1'].isin(new))]
df_new2 = result.iloc[np.where(result['2'].isin(new2))]
# df_new3 = result.iloc[np.where(result['3'].isin(new2))]
# new_dataframe = pd.concat([df_new, df_new2]).drop_duplicates()
new_dataframe = pd.concat([df_new,df_new2]).drop_duplicates()
new_dataframe = df_new.filter(['id', 'text', 'preds_y'])
new_dataframe['preds_y'] = new_dataframe['preds_y'].map({0.0: 'NOT', 1.0: 'OFF'})
new_dataframe.rename({'text': 'content', 'preds_y': 'Class'}, axis=1, inplace=True)
new_dataframe.to_csv('new_train.csv')

test_set.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

df_nw = pd.read_csv(arguments.train, sep="\t")
df_merged = df_nw.append(new_dataframe, ignore_index=True)
# how to replace this to same argument?????
df_merged.to_csv('data/new_sold.tsv', sep="\t")




