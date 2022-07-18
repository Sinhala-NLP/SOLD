import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import sklearn
import torch
import gc
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

data = pd.read_csv('/content/SOLD/data/sold_trial.tsv', sep="\t")
data = data.rename(columns={'tweet': 'text', 'subtask_a': 'labels', 'id': 'post_id'})
data = data[['text', 'labels', 'post_id']]

train, test = train_test_split(data[:100], test_size=0.2)

# Train the model
print("Started Training")

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test), args["n_fold"]))

if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
                                    use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        predictions, raw_outputs = model.predict(test_sentences)
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
                                use_cuda=torch.cuda.is_available())
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

time.sleep(5)

print_information_multi_class(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')

#TODO: Refactor above code
#TODO: Parameterize (1) train and test with rationals (2) test with rationals
gc.collect()

#TODO: Add separate condition to load and rational only
# model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args, use_cuda=torch.cuda.is_available())

list_dict, test_data = model.standaloneEval_with_rational(test_sentences, test_data = test)