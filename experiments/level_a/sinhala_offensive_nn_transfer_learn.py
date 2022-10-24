import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from offensive_nn.config.sold_config import tl_args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--lang', required=False, help='language', default="sin")  # en or sin
parser.add_argument('--tr_lang', required=False, help='transfer learn language', default="hin")  # en or sin
parser.add_argument('--algorithm', required=False, help='algorithm', default="cnn2D")  # lstm or cnn2D
parser.add_argument('--transferlearn', required=False, help='transfer learn', default=False)  # lstm or cnn2D
parser.add_argument('--tr_embeddings', required=False, help='transfer learn embeddings',
                    default=None)
arguments = parser.parse_args()


def retrieve_train_test_sets(lang="sin"):
    if lang == "en":
        train_set = pd.read_csv('data/other/other-data_sub_task_a.tsv', sep="\t")
        test_set = pd.read_csv('data/other/testset-levela.tsv', sep="\t")
        olid_test_labels = pd.read_csv('data/other/labels-levela.csv', names=['index', 'labels'])

        train_set = train_set[['text', 'labels']]
        test_set = test_set.rename(columns={'tweet': 'text'})
        test_set['labels'] = encode(olid_test_labels['labels'])

        train_set['labels'] = encode(train_set["labels"])

    elif lang == "sin":
        sold_train_file = pd.read_csv('data/SOLD_train.tsv', sep="\t")
        train = sold_train_file.rename(columns={'content': 'text', 'Class': 'labels'})

        sold_test_file = pd.read_csv('data/SOLD_test.tsv', sep="\t")
        test = sold_test_file.rename(columns={'content': 'text', 'Class': 'labels'})

        train_set = train[['text', 'labels']]
        train_set['labels'] = encode(train_set['labels'])
        test_set = test[['text', 'labels']]
        test_set['labels'] = encode(test_set['labels'])

    elif lang == "hin":
        hindi_train_file = pd.read_csv('data/other/hindi_dataset.tsv', sep="\t")
        train = hindi_train_file.rename(columns={'task_1': 'labels'})

        hindi_test_file = pd.read_csv('data/other/hasoc2019_hi_test_gold_2919.tsv', sep="\t")
        test = hindi_test_file.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

        train_set = train[['text', 'labels']]
        train_set['labels'] = encode(train_set['labels'])
        test_set = test[['text', 'labels']]
        test_set['labels'] = encode(test_set['labels'])

    return train_set, test_set


if arguments.transferlearn:
    train_tr_learn, test_tr_learn = retrieve_train_test_sets(arguments.tr_lang)
train_set, test_set = retrieve_train_test_sets(arguments.lang)

test_sentences = test_set['text'].tolist()

test_preds = np.zeros((len(test_set), tl_args["n_fold"]))
tl_args['n_fold'] = 1
for i in range(tl_args["n_fold"]):

    if arguments.transferlearn:

        full_train_set = pd.concat([train_set, train_tr_learn])
        train_set, validation_set = train_test_split(full_train_set, test_size=0.2, random_state=tl_args["manual_seed"])

        # pass a list of embeddings and combined datasets
        model = OffensiveNNModel(model_type_or_path=arguments.algorithm,
                                 embedding_model_name_or_path=arguments.model_name,
                                 train_df=train_set,
                                 args=tl_args, eval_df=validation_set,
                                 emd_file=arguments.tr_embeddings)
        tl_train_set, tl_validation_set = train_test_split(train_tr_learn, test_size=0.2,
                                                           random_state=tl_args["manual_seed"])
        model.transfer_learn_train_model(train_df=tl_train_set, eval_df=tl_validation_set)
        train_set, validation_set = train_test_split(train_set, test_size=0.2,
                                                     random_state=tl_args["manual_seed"])
        model.transfer_learn_train_model(train_df=train_set, eval_df=validation_set)
        print("Finished Training")
        model = OffensiveNNModel(model_type_or_path=tl_args["best_model_dir"])
        predictions, raw_outputs = model.predict(test_sentences)
        test_preds[:, i] = predictions
        print("Completed Fold {}".format(i))
    else:

        train_set, validation_set = train_test_split(train_set, test_size=0.2, random_state=tl_args["manual_seed"])
        model = OffensiveNNModel(model_type_or_path=arguments.algorithm,
                                 embedding_model_name_or_path=arguments.model_name,
                                 train_df=train_set,
                                 args=tl_args, eval_df=validation_set)
        model.train_model()
        print("Finished Training")
        model = OffensiveNNModel(model_type_or_path=tl_args["best_model_dir"])
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
