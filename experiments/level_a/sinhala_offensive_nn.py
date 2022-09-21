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


import numpy as np



parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="word2vec-google-news-300")
parser.add_argument('--lang', required=False, help='language', default="en")  # en or sin
parser.add_argument('--algorithm', required=False, help='algorithm', default="cnn2D")  # lstm or cnn2D
parser.add_argument('--train', required=False, help='train file', default='data/olid/olid-training-v1.0.tsv')
parser.add_argument('--test', required=False, help='test file')
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
    test = hindi_test_file.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

    # train, test = train_test_split(hindi_train_file, test_size=0.1, random_state=777)
    train_set = train[['text', 'labels']]
    train_set['labels'] = encode(train_set['labels'])
    test_set = test[['text', 'labels']]
    test_set['labels'] = encode(test_set['labels'])

test_sentences = test_set['text'].tolist()

test_preds = np.zeros((len(test_set), args["n_fold"]))

if args["evaluate_during_training"]:
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
else:
    model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name_or_path=arguments.model_name,
                                 train_df=train_set,
                                 args=args)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    # print(raw_outputs)
    confidence_df = pd.DataFrame(raw_outputs)
    test['preds'] = predictions
    predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
    predictions_df.to_csv('prediction.csv')
    confidence_df.to_csv('confidence_result1.csv')
    test['predictions'] = predictions

    df1 = pd.read_csv('prediction.csv')
    column_names = ['1', '2', '3']
    df = pd.read_csv('confidence_result1.csv', names=column_names, header=None)
    frames = [df, df1]
    result = pd.concat([df1, df], axis=1)

    new = []
    new1 = []
    new2 = []

    m1 = np.mean(df['1'])
    m2 = np.mean(df['2'])
    m3 = np.mean(df['3'])

    print(m1, m2, m3)

    l1 = np.std(df['1'])
    l2 = np.std(df['2'])
    l3 = np.std(df['3'])

    print(l1, l2, l3)

    # 2.5, 2.0, 1,5, 1, 0.5

    st1 = l1 / 2.5
    st2 = l2 / 2.5
    st3 = l3 / 2.5

    print(st1, st2, st3)

    for ix in df.index:
        e = df.loc[ix]['1']
        full = e - m1
        f = df.loc[ix]['2']
        full2 = f - m2
        g = df.loc[ix]['3']
        full3 = g - m3

        if (full > st1):
            new.append(df.loc[ix]['1'])
            # print(new)
        if (full2 > st2):
            new1.append(df.loc[ix]['2'])
            # print(new1)
        if (full3 > st3):
            new2.append(df.loc[ix]['3'])
            # print(new2)

    # print(l1)
    # print(l2)
    # print(l3)
    #
    # print(m1)
    # print(m2)
    # print(m3)

    df_new = result.iloc[np.where(result['1'].isin(new))]
    df_new2 = result.iloc[np.where(result['2'].isin(new1))]
    df_new3 = result.iloc[np.where(result['3'].isin(new2))]
    new_dataframe = pd.concat([df_new, df_new2, df_new3]).drop_duplicates()
    # new_dataframe = pd.concat([df_new,df_new2]).drop_duplicates()
    new_dataframe = new_dataframe.filter(['id', 'text', 'preds_y'])
    new_dataframe['preds_y'] = new_dataframe['preds_y'].map({0.0: 'NOT', 1.0: 'OFF'})
    new_dataframe.rename({'text': 'content', 'preds_y': 'Class'}, axis=1, inplace=True)
    new_dataframe.to_csv('new_train.csv')

    model.save_model()

    # test['predictions'] = decode(test['predictions'])

    # time.sleep(5)

    # test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

    df_nw = pd.read_csv(arguments.train, sep="\t")
    df_merged = df_nw.append(new_dataframe, ignore_index=True)
    # how to replace this to same argument?????
    df_merged.to_csv('data/new_sold.tsv', sep="\t")




test_set['predictions'] = decode(test_set['predictions'])
test_set['labels'] = decode(test_set['labels'])

print_information(test_set, "predictions", "labels")
