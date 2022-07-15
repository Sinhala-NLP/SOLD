import argparse
import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
from deepoffense.util.print_stat import print_information, print_information_multi_class

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=1)
parser.add_argument('--train', required=False, help='train file', default='data/olid/olid-training-v1.0.tsv')
parser.add_argument('--test', required=False, help='test file', default='data/olid/olid-training-v1.0.tsv')
arguments = parser.parse_args()

data = pd.read_csv(arguments.train, sep="\t")
data = data.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
train = data[['text', 'labels']]

if not arguments.test:

    # load new csv
    # new_df=pd.read_csv('data/new_train.csv')
    # df['labels'] = df['labels'].map({0.0:'OFF',1.0:'NOT OFF'})

    train, test = train_test_split(data, test_size=0.2)
    if LANGUAGE_FINETUNE:
        train_list = train['text'].tolist()
        test_list = test['text'].tolist()
        complete_list = train_list + test_list
        lm_train = complete_list[0: int(len(complete_list) * 0.8)]
        lm_test = complete_list[-int(len(complete_list) * 0.2):]

        with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
            for item in lm_train:
                f.write("%s\n" % item)

        with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
            for item in lm_test:
                f.write("%s\n" % item)

        model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
        model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"),
                          eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
        MODEL_NAME = language_modeling_args["best_model_dir"]

    # Train the model
    print("Started Training")

    train['labels'] = encode(train["labels"])
    # c
    # test['labels'] = encode(test["labels"])

    test_sentences = test['text'].tolist()
    test_preds = np.zeros((len(test), args["n_fold"]))

    MODEL_NAME = arguments.model_name
    MODEL_TYPE = arguments.model_type
    cuda_device = arguments.cuda_device

    if args["evaluate_during_training"]:
        for i in range(args["n_fold"]):
            if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
                shutil.rmtree(args['output_dir'])
            print("Started Fold {}".format(i))
            torch.cuda.set_device(cuda_device)
            model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
                                        use_cuda=torch.cuda.is_available(),
                                        cuda_device=cuda_device)  # You can set class weights by using the optional weight argument
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              accuracy=sklearn.metrics.accuracy_score)
            # model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
            #                             use_cuda=torch.cuda.is_available())

            predictions, raw_outputs = model.predict(test_sentences)
            print(predictions,raw_outputs)
            test_preds[:, i] = predictions
            print("Completed Fold {}".format(i))
        # select majority class of each instance (row)
        final_predictions = []
        for row in test_preds:
            row = row.tolist()
            final_predictions.append(int(max(set(row), key=row.count)))
        test['predictions'] = final_predictions
    else:
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
                                    use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
        model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        predictions, raw_outputs = model.predict(test_sentences)
        print(raw_outputs)
        confidence_df=pd.DataFrame(raw_outputs)
        test['preds'] = predictions
        predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
        # predictions_df = pd.DataFrame.from_dict({'y_test': test, 'predictions': predictions}).to_csv('prediction.csv')
        # need to add
        # predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
        # predictions_df.to_csv('prediction_result.csv')
        predictions_df.to_csv('prediction.csv')
        confidence_df.to_csv('confidence_result.csv')

        test['predictions'] = predictions

    model.save_model()

    test['predictions'] = decode(test['predictions'])
    # c
    test['labels'] = decode(test['labels'])

    # time.sleep(5)
    print_information_multi_class(test, "predictions", "labels")

    print_information_multi_class(test, "predictions")

    test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

else :

    train = data[['text', 'labels']]
    test= pd.read_csv(arguments.test, sep=",")

    if LANGUAGE_FINETUNE:
        train_list = train['text'].tolist()
        test_list = test['text'].tolist()
        complete_list = train_list + test_list
        lm_train = complete_list[0: int(len(complete_list) * 0.8)]
        lm_test = complete_list[-int(len(complete_list) * 0.2):]

        with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
            for item in lm_train:
                f.write("%s\n" % item)

        with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
            for item in lm_test:
                f.write("%s\n" % item)

        model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
        model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"),
                          eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
        MODEL_NAME = language_modeling_args["best_model_dir"]

    # Train the model
    print("Started Training")

    train['labels'] = encode(train["labels"])
    test_sentences = test['text'].tolist()
    test_preds = np.zeros((len(test), args["n_fold"]))

    MODEL_NAME = arguments.model_name
    MODEL_TYPE = arguments.model_type
    cuda_device = arguments.cuda_device

    if args["evaluate_during_training"]:
        for i in range(args["n_fold"]):
            if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
                shutil.rmtree(args['output_dir'])
            print("Started Fold {}".format(i))
            torch.cuda.set_device(cuda_device)
            model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
                                        use_cuda=torch.cuda.is_available(),
                                        cuda_device=cuda_device)  # You can set class weights by using the optional weight argument
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                              accuracy=sklearn.metrics.accuracy_score)
            # model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
            #                             use_cuda=torch.cuda.is_available())

            predictions, raw_outputs = model.predict(test_sentences)
            print(predictions, raw_outputs)
            test_preds[:, i] = predictions
            print("Completed Fold {}".format(i))
        # select majority class of each instance (row)
        final_predictions = []
        for row in test_preds:
            row = row.tolist()
            final_predictions.append(int(max(set(row), key=row.count)))
        test['predictions'] = final_predictions
    else:
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
                                    use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
        model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        predictions, raw_outputs = model.predict(test_sentences)
        print(raw_outputs)
        confidence_df = pd.DataFrame(raw_outputs)
        test['preds'] = predictions
        predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
        predictions_df.to_csv('prediction.csv')
        confidence_df.to_csv('confidence_result.csv')
        test['predictions'] = predictions

        df1 = pd.read_csv('prediction.csv')
        column_names = ['1', '2', '3']
        df = pd.read_csv('confidence_result.csv', names=column_names, header=None)
        frames = [df, df1]
        result = pd.concat([df1, df], axis=1)

        new = []
        new1 = []
        new2 = []

        m1=np.mean(df['1'])
        m2=np.mean(df['2'])
        m3=np.mean(df['3'])

        for ix in df.index:
            e = df.loc[ix]['1']
            f = df.loc[ix]['2']
            g = df.loc[ix]['3']
            full = e - m1
            full2 = f - m2
            full3 = g - m3

        l1 = np.std(df['1'])
        l2 = np.std(df['2'])
        l3 = np.std(df['3'])

        if (full < l1/2):
            new.append(df.loc[ix]['1'])
        elif (full2 < l2/2):
            new1.append(df.loc[ix]['2'])
        elif (full3 < l3/2):
            new2.append(df.loc[ix]['3'])

        df_new = result.iloc[np.where(result['1'].isin(new))]
        df_new2 = result.iloc[np.where(result['2'].isin(new1))]
        df_new3 = result.iloc[np.where(result['3'].isin(new2))]
        new_dataframe = pd.concat([df_new, df_new2, df_new3]).drop_duplicates()
        new_dataframe = new_dataframe.filter(['id', 'text', 'preds_y'])
        new_dataframe['preds_y'] = new_dataframe['preds_y'].map({0.0: 'NOT', 1.0: 'OFF'})
        new_dataframe.rename({'text': 'tweet', 'preds_y': 'subtask_a'}, axis=1, inplace=True)
        new_dataframe.to_csv('new_train.csv')

    model.save_model()

    test['predictions'] = decode(test['predictions'])

    # time.sleep(5)
    test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

    df_nw=pd.read_csv(arguments.train, sep="\t")
    df_merged = df_nw.append(new_dataframe, ignore_index=True)
    # how to replace this to same argument?????
    df_merged.to_csv('/content/SOLD/data/new_sold.tsv', sep="\t")


    df_merged