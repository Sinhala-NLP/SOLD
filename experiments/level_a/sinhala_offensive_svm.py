import argparse

import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from deepoffense.util.label_converter import decode, encode
from offensive_nn.util.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--train', required=False, help='train file', default='data/SOLD_train.tsv')
parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
parser.add_argument('--lang', required=False, help='language', default="sin")  # sin
arguments = parser.parse_args()

# load datafiles related to different languages
trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

# trn_data, tst_data = train_test_split(trn_data, test_size=0.1)
if arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'content': 'text', 'Class': 'labels'})
    tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})

train = trn_data[['text', 'labels']]
train['labels'] = encode(train["labels"])
test = tst_data[['text', 'labels']]
test['labels'] = encode(test['labels'])

train_list = train['text'].tolist()
test_list = test['text'].tolist()

all_text = train_list + test_list


def flatten_words(list1d, get_unique=False):
    qa = [s.split() for s in list1d]

    if get_unique:
        y = sorted(list(set([w for sent in qa for w in sent])))
        return y
    else:
        e = [w for sent in qa for w in sent]
        return e


# create vocabulary based on the size of data
vocab = flatten_words(all_text, get_unique=True)
tfidf = TfidfVectorizer(vocabulary=vocab)  # max_features=5000
training_matrix = tfidf.fit_transform(train['text'])
test_matrix = tfidf.fit_transform(test['text'])

model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
model.fit(training_matrix, train['labels'])
predictions = model.predict(test_matrix)
print("Completed Predictions")

test['predictions'] = predictions
test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

print_information(test, "predictions", "labels")
