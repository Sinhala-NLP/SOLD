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
    train_set = pd.read_csv('data/other/other-data_sub_task_a.tsv', sep="\t")
    test_set = pd.read_csv('data/other/testset-levela.tsv', sep="\t")
    olid_test_labels = pd.read_csv('data/other/labels-levela.csv', names=['index', 'labels'])

    train_set = train_set[['text', 'labels']]
    test_set = test_set.rename(columns={'tweet': 'text'})
    test_set['labels'] = encode(olid_test_labels['labels'])

    train_set['labels'] = encode(train_set["labels"])

elif arguments.lang == "sin":
    sold_train_file = pd.read_csv('data/SOLD_train.tsv', sep="\t")
    train = sold_train_file.rename(columns={'content': 'text', 'Class': 'labels'})

    sold_test_file = pd.read_csv('data/SOLD_test.tsv', sep="\t")
    test = sold_test_file.rename(columns={'content': 'text', 'Class': 'labels'})

    # train, test = train_test_split(sold_train_file, test_size=0.1, random_state=777)

    train_set = train[['text', 'labels']]
    train_set['labels'] = encode(train_set['labels'])
    test_set = test[['text', 'labels']]
    test_set['labels'] = encode(test_set['labels'])


elif arguments.lang == "hin":
    hindi_train_file = pd.read_csv('data/other/hindi_dataset.tsv', sep="\t")
    train = hindi_train_file.rename(columns={'task_1': 'labels'})

    hindi_test_file = pd.read_csv('data/other/hasoc2019_hi_test_gold_2919.tsv', sep="\t")
    test = hindi_test_file.rename(columns={'task_1': 'labels'})

    # train, test = train_test_split(hindi_train_file, test_size=0.1, random_state=777)
    train_set = train[['text', 'labels']]
    train_set['labels'] = encode(train_set['labels'])
    test_set = test[['text', 'labels']]
    test_set['labels'] = encode(test_set['labels'])

test_sentences = test_set['text'].tolist()

test_preds = np.zeros((len(test_set), args["n_fold"]))

for i in range(args["n_fold"]):
    train_set, validation_set = train_test_split(train_set, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name_or_path=arguments.model_name,
                             train_df=train_set,
                             args=args, eval_df=validation_set)
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

test_set['predictions'] = final_predictions
test_set['predictions'] = decode(test_set['predictions'])
test_set['labels'] = decode(test_set['labels'])

print_information(test_set, "predictions", "labels")
