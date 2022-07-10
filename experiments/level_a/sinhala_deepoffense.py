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

train, test = train_test_split(data, test_size=0.2)
# c
# test= pd.read_csv(arguments.test, sep=",")




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
test['labels'] = encode(test["labels"])

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
test.to_csv()
# c
test['labels'] = decode(test['labels'])

# time.sleep(5)

# c
print_information_multi_class(test, "predictions", "labels")



# print_information_multi_class(test, "predictions")

test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

def calculate_std(acc):
    label1 = np.std(confidence_df['1'])
    label2 = np.std(confidence_df['2'])
    label3 = np.std(confidence_df['3'])



