import argparse
import os
import pandas as pd
import shutil
import sklearn
import statistics
import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from deepoffense.util.print_stat import print_information
from experiments.sentence_level.deepoffense_config import TEMP_DIRECTORY, sinhala_args, hindi_args, SEED, \
    RESULT_FILE, english_args, cmcs_args

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--augment', required=False, help='augment', default="false")
parser.add_argument('--std', required=False, help='standard deviation', default="0.01")
parser.add_argument('--augment_type', required=False, help='type of the data augmentation', default="off")
parser.add_argument('--transfer', required=False, help='transfer learning', default="false")
parser.add_argument('--transfer_language', required=False, help='transfer learning', default="hi")
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

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

if arguments.transfer == "true" and arguments.transfer_language == "hi":
    if os.path.exists(hindi_args['output_dir']) and os.path.isdir(hindi_args['output_dir']):
        shutil.rmtree(hindi_args['output_dir'])

    hindi_train = pd.read_csv("data/other/hindi_dataset.tsv", sep="\t")
    hindi_train = hindi_train.rename(columns={'task_1': 'labels'})
    hindi_train = hindi_train[['text', 'labels']]
    hindi_train['labels'] = hindi_train['labels'].replace(['HOF'], 'OFF')
    hindi_train['labels'] = encode(hindi_train["labels"])

    hindi_test = pd.read_csv("data/other/hasoc2019_hi_test_gold_2919.tsv", sep="\t")
    hindi_test = hindi_test.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})
    hindi_test = hindi_test[['text', 'labels']]
    hindi_test['labels'] = hindi_test['labels'].replace(['HOF'], 'OFF')
    hindi_test['labels'] = encode(hindi_test["labels"])

    hindi_test_sentences = hindi_test['text'].tolist()

    hindi_train_df, hindi_eval_df = train_test_split(hindi_train, test_size=0.1, random_state=SEED)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=hindi_args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    model.train_model(hindi_train_df, eval_df=hindi_eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    hindi_predictions, hindi_raw_outputs = model.predict(hindi_test_sentences)
    hindi_test['predictions'] = hindi_predictions
    hindi_test['predictions'] = decode(hindi_test['predictions'])
    hindi_test['labels'] = decode(hindi_test['labels'])

    # time.sleep(5)
    print("Hindi Results")
    print_information(hindi_test, "predictions", "labels")
    MODEL_NAME = hindi_args['best_model_dir']

if arguments.transfer == "true" and arguments.transfer_language == "en":
    if os.path.exists(english_args['output_dir']) and os.path.isdir(english_args['output_dir']):
        shutil.rmtree(english_args['output_dir'])

    english_train = pd.read_csv("data/other/olid-training-v1.0.tsv", sep="\t")
    english_train = english_train.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
    english_train = english_train[['text', 'labels']]
    english_train['labels'] = encode(english_train["labels"])

    english_test = pd.read_csv("data/other/testset-levela.tsv", sep="\t")
    english_labels = pd.read_csv("data/other/labels-levela.csv", names=['id', 'labels'])
    english_test["labels"] = english_labels["labels"]
    english_test = english_test.rename(columns={'tweet': 'text'})
    english_test = english_test[['text', 'labels']]
    english_test['labels'] = encode(english_test["labels"])

    english_test_sentences = english_test['text'].tolist()

    english_train_df, english_eval_df = train_test_split(english_train, test_size=0.1, random_state=SEED)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=english_args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    model.train_model(english_train_df, eval_df=english_eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    english_predictions, english_raw_outputs = model.predict(english_test_sentences)
    english_test['predictions'] = english_predictions
    english_test['predictions'] = decode(english_test['predictions'])
    english_test['labels'] = decode(english_test['labels'])

    # time.sleep(5)
    print("English Results")
    print_information(english_test, "predictions", "labels")
    MODEL_NAME = english_args['best_model_dir']

if arguments.transfer == "true" and arguments.transfer_language == "si":
    if os.path.exists(cmcs_args['output_dir']) and os.path.isdir(cmcs_args['output_dir']):
        shutil.rmtree(cmcs_args['output_dir'])

    ccms = pd.read_csv("data/other/ccms-sentence-level-annotation.csv", sep=",")
    ccms = ccms.rename(columns={'Sentence': 'text', 'Hate_speech': 'labels', })
    ccms = ccms[['text', 'labels']]
    ccms['labels'] = ccms['labels'].replace(['Not offensive', 'Abusive', 'Hate-Inducing'], ['NOT', 'OFF', 'OFF'])
    ccms['labels'] = encode(ccms["labels"])

    ccms_train_df, ccms_test_df = train_test_split(ccms, test_size=0.25, random_state=SEED)
    ccms_test_sentences = ccms_test_df['text'].tolist()
    ccms_train, ccms_eval = train_test_split(ccms_train_df, test_size=0.2, random_state=SEED)

    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=cmcs_args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    model.train_model(ccms_train, eval_df=ccms_eval, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    ccms_predictions, ccms_raw_outputs = model.predict(ccms_test_sentences)
    ccms_test_df['predictions'] = ccms_predictions
    ccms_test_df['predictions'] = decode(ccms_test_df['predictions'])
    ccms_test_df['labels'] = decode(ccms_test_df['labels'])

    # time.sleep(5)
    print("CCMS Results")
    print_information(ccms_test_df, "predictions", "labels")
    MODEL_NAME = cmcs_args['best_model_dir']

if sinhala_args["evaluate_during_training"]:
    if os.path.exists(sinhala_args['output_dir']) and os.path.isdir(sinhala_args['output_dir']):
        shutil.rmtree(sinhala_args['output_dir'])
    torch.cuda.set_device(cuda_device)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=sinhala_args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
    if arguments.augment == "true":
        print("Downloading SemiSOLD")
        semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
        std = float(arguments.std)
        augment_type = arguments.augment_type
        complete_df = []
        for index, row in semi_sold.iterrows():
            model_scores = [row['xlmr'], row['xlmt'], row['sinbert']]
            model_std = statistics.stdev(model_scores)
            if model_std < std:
                model_average = statistics.mean(model_scores)
                label = "OFF" if model_average > 0.5 else "NOT"
                complete_df.append([row['text'], label])

        df = pd.DataFrame(complete_df, columns=["text", "labels"])
        df['labels'] = encode(df["labels"])
        if augment_type == "off":
            filtered_df = df.loc[df['labels'] == 1]
        else:
            filtered_df = df

        print("Augmenting {} records".format(filtered_df.shape[0]))
        train_df = train_df.append(filtered_df)
        train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)

    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions
else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=sinhala_args,
                                use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
    if arguments.augment == "true":
        print("Downloading SemiSOLD")
        semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
        std = float(arguments.std)
        augment_type = arguments.augment_type
        complete_df = []
        for index, row in semi_sold.iterrows():
            model_scores = [row['xlmr'], row['xlmt'], row['sinbert']]
            model_std = statistics.stdev(model_scores)
            if model_std < std:
                model_average = statistics.mean(model_scores)
                label = "OFF" if model_average > 0.5 else "NOT"
                complete_df.append([row['text'], label])

        df = pd.DataFrame(complete_df, columns=["text", "labels"])
        df['labels'] = encode(df["labels"])
        if augment_type == "off":
            filtered_df = df.loc[df['labels'] == 1]
        else:
            filtered_df = df

        train_df = train.append(filtered_df)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions

model.save_model()

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

# time.sleep(5)

print_information(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
