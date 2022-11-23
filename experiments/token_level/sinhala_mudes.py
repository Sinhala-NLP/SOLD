import argparse
import json
import pandas as pd
import torch
import numpy as np
import os
import shutil
import ast

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from deepoffense.classification import ClassificationModel
from scipy.special import softmax
from datasets import Dataset
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
from experiments.sentence_level.deepoffense_config import english_args
from experiments.token_level.mudes_config import sinhala_args, hatex_args
from sklearn.model_selection import train_test_split

from experiments.token_level.print_stat import print_information
from mudes.algo.mudes_model import MUDESModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--transfer', required=False, help='transfer learning', default="false")
parser.add_argument('--transfer_language', required=False, help='transfer learning', default="hatex")
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)


if arguments.transfer == "true" and arguments.transfer_language == "hatex":
    if os.path.exists(hatex_args['output_dir']) and os.path.isdir(hatex_args['output_dir']):
        shutil.rmtree(hatex_args['output_dir'])
    hatex_train = pd.read_csv("data/other/hatexplain_train.csv")
    hatex_test = pd.read_csv("data/other/hatexplain_test.csv")

    hatex_train_token_df = []
    for index, row in hatex_train.iterrows():
        tokens = ast.literal_eval(row["post_tokens"])
        labels = ast.literal_eval(row["toxic_tokens"])
        for token, label in zip(tokens, labels):
            hatex_train_token_df.append([index, token, label])

    hatex_train_data = pd.DataFrame(
        hatex_train_token_df, columns=["sentence_id", "words", "labels"])

    hatex_train_data['labels'] = hatex_train_data['labels'].replace([0, 1], ['NOT', 'OFF'])

    hatex_test_token_df = []
    hatex_test_sentences = []
    for index, row in hatex_test.iterrows():
        tokens = ast.literal_eval(row["post_tokens"])
        labels = ast.literal_eval(row["toxic_tokens"])
        hatex_test_sentences.append(' '.join(token for token in tokens))
        for token, label in zip(tokens, labels):
            hatex_test_token_df.append([index, token, label])

    hatex_test_data = pd.DataFrame(
        hatex_test_token_df, columns=["sentence_id", "words", "labels"])

    hatex_test_data['labels'] = hatex_test_data['labels'].replace([0, 1], ['NOT', 'OFF'])

    tags = hatex_train_data['labels'].unique().tolist()
    model = MUDESModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=hatex_args)
    train_df, eval_df = train_test_split(hatex_train_data, test_size=0.1, shuffle=False)
    model.train_model(train_df, eval_df=eval_df)
    hatex_predictions, hatex_raw_outputs = model.predict(hatex_test_sentences)
    hatex_final_predictions = []
    for hatex_prediction in hatex_predictions:
        hatex_raw_prediction = []
        for word_prediction in hatex_prediction:
            for key, value in word_prediction.items():
                hatex_raw_prediction.append(value)
        hatex_final_predictions.append(hatex_raw_prediction)

    converted_predictions = []
    for final_prediction, sentence in zip(hatex_final_predictions, hatex_test_sentences):
        final_prediction += (len(sentence.split()) - len(final_prediction)) * ["NOT"]
        converted_predictions.append(final_prediction)

    flat_predictions = [j for sub in converted_predictions for j in sub]
    hatex_test_data["predictions"] = flat_predictions
    print_information(hatex_test_data, "labels", "predictions")
    MODEL_NAME = hatex_args['best_model_dir']


sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

train_sentence_id = 0
train_token_df = []
for index, row in sold_train.iterrows():
    tokens = row["tokens"].split()
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            processed_row = [train_sentence_id, token, 0]
            train_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            processed_row = [train_sentence_id, token, label]
            train_token_df.append(processed_row)
    train_sentence_id = train_sentence_id + 1

train_data = pd.DataFrame(
    train_token_df, columns=["sentence_id", "words", "labels"])

train_data['labels'] = train_data['labels'].replace([0, 1], ['NOT', 'OFF'])

test_sentence_id = 0
test_token_df = []
for index, row in sold_test.iterrows():
    tokens = row["tokens"].split()
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            processed_row = [test_sentence_id, token, 0]
            test_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            processed_row = [test_sentence_id, token, label]
            test_token_df.append(processed_row)
    test_sentence_id = test_sentence_id + 1

test_data = pd.DataFrame(
    test_token_df, columns=["sentence_id", "words", "labels"])

test_data['labels'] = test_data['labels'].replace([0, 1], ['NOT', 'OFF'])

tags = train_data['labels'].unique().tolist()
model = MUDESModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=sinhala_args)
train_df, eval_df = train_test_split(train_data, test_size=0.1, shuffle=False)
model.train_model(train_df, eval_df=eval_df)
predictions, raw_outputs = model.predict(sold_test["tokens"].tolist())
final_predictions = []
for prediction in predictions:
    raw_prediction = []
    for word_prediction in prediction:
        for key, value in word_prediction.items():
            raw_prediction.append(value)
    final_predictions.append(raw_prediction)

sentences = sold_test["tokens"].tolist()
converted_predictions = []
for final_prediction, sentence in zip(final_predictions, sentences):
    final_prediction += (len(sentence.split()) - len(final_prediction)) * ["NOT"]
    converted_predictions.append(final_prediction)

flat_predictions = [j for sub in converted_predictions for j in sub]
test_data["predictions"] = flat_predictions
print_information(test_data, "labels", "predictions")




