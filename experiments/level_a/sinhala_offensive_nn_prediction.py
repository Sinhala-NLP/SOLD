import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.config.sold_config import args
from scipy.special import softmax
import numpy as np


# load arguments
parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--lang', required=False, help='language', default="en")  # en or sin
parser.add_argument('--algorithm', required=False, help='algorithm', default="cnn2D")  # lstm or cnn2D
parser.add_argument('--train', required=False, help='train file', default='data/olid/olid-training-v1.0.tsv')
parser.add_argument('--test', required=False, help='test file')
parser.add_argument('--sdvalue', required=False, help='standard deviation', default=0.01)
arguments = parser.parse_args()

# load datafiles related to different languages
trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

if arguments.lang == "en":
    trn_data, tst_data = train_test_split(trn_data, test_size=0.1)
elif arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'label': 'labels'})
    tst_data = tst_data.rename(columns={'label': 'labels'})
elif arguments.lang == "hin":
    trn_data = trn_data.rename(columns={'task_1': 'labels'})
    tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

# process the datafiles
# Train the model
print("Started Training")
train_set = trn_data[['text', 'labels']]
train_set['labels'] = encode(train_set['labels'])
test_set = tst_data[['text']]
test_sentences = test_set['text'].tolist()

test_preds = np.zeros((len(test_set), args["n_fold"]))

for i in range(args["n_fold"]):
    train_set, validation_set = train_test_split(train_set, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(embedding_model_name_or_path=arguments.model_name, model_type_or_path=arguments.algorithm,
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

# get confidence score and predictions
confidence_df = pd.DataFrame(probs)
test_set['label'] = predictions
test_set.to_csv('prediction.csv')
confidence_df.to_csv('confidence_result1.csv', index=False)
test_set['predictions'] = predictions
df1 = pd.read_csv('prediction.csv')

column_names = ['1', '2']
df = pd.read_csv('confidence_result1.csv', names=column_names, header=None)
frames = [df, df1]
result = pd.concat([df1, df], axis=1)
result.to_csv('one_prediction.csv')
new = []
new1 = []
new2 = []

# get the mean value of the labels
m1 = np.mean(df['1'])
m2 = np.mean(df['2'])

# Adjustable standard deviation value
l1 = float(arguments.sdvalue)
# get all the offensive and not offensive posts from the dataset
df_group_posts = result.groupby('label')
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
new_dataframe = pd.concat([df_new,df_new2]).drop_duplicates()
new_dataframe = df_new.filter(['id', 'text', 'label'])
new_dataframe['label'] = new_dataframe['label'].map({0.0: 'NOT', 1.0: 'OFF'})
# new_dataframe.rename({'text': 'text', 'preds': 'label'}, axis=1, inplace=True)
new_dataframe.to_csv('new_train.csv')

# create new dataframe after filtering the rows
df_nw = pd.read_csv(arguments.train, sep="\t")
df_nw = df_nw [["text","label"]]
df_merged = df_nw.append(new_dataframe, ignore_index=True)
df_merged.to_csv('data/new_sold.tsv', sep="\t")




