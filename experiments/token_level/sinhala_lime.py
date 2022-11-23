import argparse
import json
import pandas as pd
import torch
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from deepoffense.classification import ClassificationModel
from scipy.special import softmax
from datasets import Dataset
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
from experiments.sentence_level.deepoffense_config import sinhala_args
from experiments.token_level.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="sinhala-nlp/xlm-t-sold-si")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))


def _sinhala_tokenizer(text):
    return text.split()


def _predict_probabilities(test_sentences):
    predictions, raw_outputs = model.predict(test_sentences)
    probabilities = softmax(raw_outputs, axis=1)
    return probabilities


sold_train = sold_train.loc[sold_train['label'] == "OFF"]
sold_test = sold_test.loc[sold_test['label'] == "OFF"]

sold_train = sold_train.head(20)
sold_test = sold_test.head(20)


model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=sinhala_args, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
explainer = LimeTextExplainer(split_expression=_sinhala_tokenizer, class_names=["NOT", "OFF"])

train_sentence_id = 0
train_token_df = []
for index, row in sold_train.iterrows():
    exp = explainer.explain_instance(row["tokens"], _predict_probabilities, num_features=200)
    explanations = exp.as_list()
    tokens = row["tokens"].split()
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [train_sentence_id, token, 0, explanation[1]]
                    train_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [train_sentence_id, token, label, explanation[1]]
                    train_token_df.append(processed_row)
    train_sentence_id = train_sentence_id + 1

train_data = pd.DataFrame(
    train_token_df, columns=["sentence_id", "words", "labels", "explanations"])

test_sentence_id = 0
test_token_df = []
for index, row in sold_test.iterrows():
    exp = explainer.explain_instance(row["tokens"], _predict_probabilities, num_features=200)
    explanations = exp.as_list()
    tokens = row["tokens"].split()
    labels = json.loads(row["rationales"])
    if len(labels) == 0:
        for token in tokens:
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [test_sentence_id, token, 0, explanation[1]]
                    test_token_df.append(processed_row)
    else:
        for token, label in zip(tokens, labels):
            for explanation in explanations:
                if token == explanation[0]:
                    processed_row = [test_sentence_id, token, label, explanation[1]]
                    test_token_df.append(processed_row)
    test_sentence_id = test_sentence_id + 1

train_data = pd.DataFrame(train_token_df, columns=["sentence_id", "words", "labels", "explanations"])
test_data = pd.DataFrame(test_token_df, columns=["sentence_id", "words", "labels", "explanations"])


X = np.array(train_data['explanations'].tolist()).reshape(-1, 1)
Y = np.array(train_data['labels'].tolist())

clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X, Y)
predictions = clf.predict(np.array(test_data['explanations'].tolist()).reshape(-1, 1))

test_data["predictions"] = predictions
print_information(test_data, "labels", "predictions")


