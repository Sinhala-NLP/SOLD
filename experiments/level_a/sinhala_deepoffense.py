import argparse
import numpy as np
import pandas as pd
import statistics
import os
import shutil
import sklearn
import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from experiments.level_a.deepoffense_config import TEMP_DIRECTORY, SUBMISSION_FOLDER, sinhala_args, SEED, RESULT_FILE
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
parser.add_argument('--augment', required=False, help='augment', default="false")
parser.add_argument('--std', required=False, help='standard deviation', default="0.01")
parser.add_argument('--augment_type', required=False, help='tyoe of the data augmentation', default="off")
# parser.add_argument('--lang', required=False, help='language', default="sin")  # en or sin or hin
arguments = parser.parse_args()

# trn_data = pd.read_csv(arguments.train, sep="\t")
# tst_data = pd.read_csv(arguments.test, sep="\t")

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))
# if arguments.lang == "en":
#     trn_data, tst_data = train_test_split(trn_data, test_size=0.1)


trn_data = sold_train.rename(columns={'label': 'labels'})
tst_data = sold_test.rename(columns={'label': 'labels'})

# elif arguments.lang == "hin":
#     trn_data = trn_data.rename(columns={'task_1': 'labels'})
#     tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]

# Train the model
print("Started Training")

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)

if sinhala_args["evaluate_during_training"]:
    if os.path.exists(sinhala_args['output_dir']) and os.path.isdir(sinhala_args['output_dir']):
        shutil.rmtree(sinhala_args['output_dir'])
    torch.cuda.set_device(cuda_device)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=sinhala_args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
    print(arguments.augment)
    if arguments.augment == "true":
        print("Downloading SemiSOLD")
        semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
        std = float(arguments.std)
        off = arguments.augment_type
        complete_df = []
        for index, row in semi_sold.iterrows():
            model_scores = [row['xlmr'], row['xlmt'], row['sinbert']]
            model_std = statistics.stdev(model_scores)
            if model_std < std:
                model_average = statistics.mean(model_scores)
                label = "OFF" if model_average > 0.5 else "NOT"
                complete_df.append([row['text'], label])

        df = pd.DataFrame(complete_df, columns=["text", "labels"])
        if off == "true":
            filtered_df = df.loc[df['labels'] == "OFF"]
        else:
            filtered_df = df

        filtered_df['labels'] = encode(filtered_df["labels"])
        train_df = pd.concat(train_df, filtered_df)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions
else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=sinhala_args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
    if arguments.augment == "true":
        semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
        std = float(arguments.std)
        off = arguments.augment_type
        complete_df = []
        for index, row in semi_sold.iterrows():
            model_scores = [row['xlmr'], row['xlmt'], row['sinbert']]
            model_std = statistics.stdev(model_scores)
            if model_std < std:
                model_average = statistics.mean(model_scores)
                label = "OFF" if model_average > 0.5 else "NOT"
                complete_df.append([row['text'], label])

        df = pd.DataFrame(complete_df, columns=["text", "labels"])
        if off == "true":
            filtered_df = df.loc[df['labels'] == "OFF"]
        else:
            filtered_df = df

        filtered_df['labels'] = encode(filtered_df["labels"])
        train = pd.concat(train, filtered_df)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions

model.save_model()

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

# time.sleep(5)

print_information(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
