import argparse
import numpy as np
import pandas as pd
from experiments.level_a.svm_config.sold_config import args
from deepoffense.util.label_converter import decode, encode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/SOLD_train.tsv')
parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
parser.add_argument('--lang', required=False, help='language', default="sin")  # en or sin or hin
parser.add_argument('--sdvalue', required=False, help='language', default=0.01)
arguments = parser.parse_args()

# load datafiles related to different languages
trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

# trn_data, tst_data = train_test_split(trn_data, test_size=0.1)
if arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'label': 'labels'})
    tst_data = tst_data.rename(columns={'label': 'labels'})


train = trn_data[['text', 'labels']]
train['labels'] = encode(train["labels"])
test = tst_data[['text']]
# test['labels'] = encode(test['labels'])

# convert words into tfidf
# process the datafiles

train_list = train['text'].tolist()
test_list = test['text'].tolist()

test_preds = np.zeros((len(test), args["n_fold"]))
all_text = train_list + test_list

def flatten_words(list1d, get_unique=False):
    wordlist = [s.split() for s in list1d]
    if get_unique:
        u_list = sorted(list(set([w for sent in wordlist for w in sent])))
        return u_list
    else:
        n_list = [w for sent in wordlist for w in sent ]
        return n_list

# create vocabulary based on the size of data
vocab = flatten_words(all_text, get_unique=True)
tfidf = TfidfVectorizer(vocabulary=vocab) #max_features=5000
training_matrix = tfidf.fit_transform(train['text'])
test_matrix = tfidf.fit_transform(test['text'])

print(training_matrix)

for i in range(args["n_fold"]):
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
    model.fit(training_matrix, train['labels'])
    predictions = model.predict(test_matrix)
    probs = model.predict_proba(test_matrix)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

test['predictions'] = final_predictions

# get confidence score and predictions
confidence_df = pd.DataFrame(probs)
test['preds'] = predictions
test.to_csv('prediction.csv')
confidence_df.to_csv('confidence_result1.csv', index=False)
test['predictions'] = predictions
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
df_group_posts = result.groupby('preds')
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
new_dataframe = df_new.filter(['id', 'text', 'preds'])
new_dataframe['preds'] = new_dataframe['preds'].map({0.0: 'NOT', 1.0: 'OFF'})
new_dataframe.rename({'text': 'text', 'preds': 'labels'}, axis=1, inplace=True)
new_dataframe.to_csv('new_train.csv')

# create new dataframe after filtering the rows
df_nw = pd.read_csv(arguments.train, sep="\t")
df_merged = df_nw.append(new_dataframe, ignore_index=True)
df_merged.to_csv('data/new_sold.tsv', sep="\t")