import argparse
import os

import numpy as np
import pandas as pd
import torch

from deepoffense.classification import ClassificationModel
from deepoffense.common.deepoffense_config import TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    args, RESULT_FILE
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
parser.add_argument('--test', required=False, help='train file', default='data/SOLD_test.tsv')
arguments = parser.parse_args()

tst_data = pd.read_csv(arguments.train, sep="\t")
tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})
test = tst_data[['text', 'labels']]

# Train the model
print("Started Prediction")

test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test), args["n_fold"]))

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = arguments.cuda_device
torch.cuda.set_device(cuda_device)

model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                            use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
predictions, raw_outputs = model.predict(test_sentences)
test['predictions'] = predictions

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

# time.sleep(5)

print_information(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
