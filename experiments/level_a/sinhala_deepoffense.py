import argparse
import os
import shutil

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from deepoffense.util.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/SOLD_train.tsv')
parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
parser.add_argument('--lang', required=False, help='language', default="sin")  # en or sin or hin
arguments = parser.parse_args()

trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

if arguments.lang == "en":
    trn_data, tst_data = train_test_split(trn_data, test_size=0.1)

elif arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'content': 'text', 'Class': 'labels'})
    tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})

elif arguments.lang == "hin":
    trn_data = trn_data.rename(columns={'task_1': 'labels'})
    tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]


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
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test), args["n_fold"]))

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)

if args["evaluate_during_training"]:
    if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
        shutil.rmtree(args['output_dir'])
    torch.cuda.set_device(cuda_device)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)  # You can set class weights by using the optional weight argument
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions
else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions

model.save_model()

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

# time.sleep(5)

print_information(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
