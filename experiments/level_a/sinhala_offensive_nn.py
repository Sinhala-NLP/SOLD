import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from offensive_nn.config.sold_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

import numpy as np

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="word2vec-google-news-300")
parser.add_argument('--lang', required=False, help='language', default="en")  # en or sin
parser.add_argument('--algorithm', required=False, help='algorithm', default="cnn2D")  # lstm or cnn2D
arguments = parser.parse_args()

if arguments.lang == "en":
    olid_train = pd.read_csv('data/olid/olid-data_sub_task_a.tsv', sep="\t")
    olid_test = pd.read_csv('data/olid/testset-levela.tsv', sep="\t")
    olid_test_labels = pd.read_csv('data/olid/labels-levela.csv', names=['index', 'labels'])

    olid_train = olid_train[['text', 'labels']]
    olid_test = olid_test.rename(columns={'tweet': 'text'})
    olid_test['labels'] = encode(olid_test_labels['labels'])

    olid_train['labels'] = encode(olid_train["labels"])
    test_sentences = olid_test['text'].tolist()
elif arguments.lang == "sin":
    sold_file = pd.read_csv('data/sold_trial.tsv', sep="\t")
    sold_file = sold_file.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})

    train, test = train_test_split(sold_file, test_size=0.1, random_state=777)
    olid_train = train[['text', 'labels']]
    olid_train['labels'] = encode(olid_train['labels'])
    olid_test = test[['text', 'labels']]
    olid_test['labels'] = encode(olid_test['labels'])

    test_sentences = olid_test['text'].tolist()

elif arguments.lang == "hin":
    sold_file = pd.read_csv('data/hin-data_sub_task_a.tsv', sep="\t")
    sold_file = sold_file.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})

    train, test = train_test_split(sold_file, test_size=0.1, random_state=777)
    olid_train = train[['text', 'labels']]
    olid_train['labels'] = encode(olid_train['labels'])
    olid_test = test[['text', 'labels']]
    olid_test['labels'] = encode(olid_test['labels'])

    test_sentences = olid_test['text'].tolist()

test_preds = np.zeros((len(olid_test), args["n_fold"]))

for i in range(args["n_fold"]):
    olid_train, olid_validation = train_test_split(olid_train, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name_or_path=arguments.model_name,
                             train_df=olid_train,
                             args=args, eval_df=olid_validation)
    model.train_model()
    print("Finished Training")
    model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
    predictions, raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

olid_test['predictions'] = final_predictions
olid_test['predictions'] = decode(olid_test['predictions'])
olid_test['labels'] = decode(olid_test['labels'])

print_information(olid_test, "predictions", "labels")
