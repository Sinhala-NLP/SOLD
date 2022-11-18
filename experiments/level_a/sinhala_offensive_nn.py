import argparse
from datasets import Dataset
from datasets import load_dataset

import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.level_a.offensivenn_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

import numpy as np

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--model_type', required=False, help='model type', default="cnn2D")  # lstm or cnn2D
arguments = parser.parse_args()

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

trn_data = sold_train.rename(columns={'label': 'labels'})
tst_data = sold_test.rename(columns={'label': 'labels'})

# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]


train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()


train_df, eval_df = train_test_split(train, test_size=0.1, random_state=args["manual_seed"])


model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name_or_path=arguments.model_name,
                             train_df=train_df,
                             args=args, eval_df=eval_df)
model.train_model()
print("Finished Training")
model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
predictions, raw_outputs = model.predict(test_sentences)

test['predictions'] = predictions
test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

print_information(test_set, "predictions", "labels")
