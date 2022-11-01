import argparse
import os
import numpy as np
import pandas as pd

from deepoffense.classification import ClassificationModel
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from offensive_nn.util.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/abc.tsv')
parser.add_argument('--test', required=False, help='test file', default='data/aaa.tsv')
parser.add_argument('--lang', required=False, help='language', default="sin")  # en or sin or hin
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
# convert words into tfidf
# process the datafiles

train_list = train['text'].tolist()
test_list = test['text'].tolist()

test_preds = np.zeros((len(test), args["n_fold"]))
all_text = train_list + test_list

def flatten_words(list1d, get_unique=False):
    qa = [s.split() for s in list1d]
    if get_unique:
        y = sorted(list(set([w for sent in qa for w in sent])))
        return y
    else:
        e = [w for sent in qa for w in sent ]
        return e

# create vocabulary based on the size of data
vocab = flatten_words(all_text, get_unique=True)
tfidf = TfidfVectorizer(vocabulary=vocab) #max_features=5000
training_matrix = tfidf.fit_transform(train['text'])
test_matrix = tfidf.fit_transform(test['text'])

print(training_matrix)

for i in range(args["n_fold"]):
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(training_matrix, train['labels'])
    predictions = model.predict(test_matrix)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

test['predictions'] = final_predictions
print(final_predictions)
print(test['labels'])
test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

print_information(test, "predictions", "labels")