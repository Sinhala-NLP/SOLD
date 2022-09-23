import argparse
import gc
import os
import shutil

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from scipy.special import softmax

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

# if (arguments.test is None):
#     trn_data = pd.read_csv(arguments.train, sep=",")
#     # tst_data = pd.read_csv(arguments.test, sep="\t")
#
#     if arguments.lang == "en":
#         trn_data, tst_data = train_test_split(trn_data, test_size=0.1)
#
#     elif arguments.lang == "sin":
#         trn_data, tst_data = train_test_split(trn_data, test_size=0.1)
#         # tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})
#
#     elif arguments.lang == "hin":
#         trn_data = trn_data.rename(columns={'task_1': 'labels'})
#         # tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})
#
#     # load training data
#     train = trn_data[['text', 'labels']]
#     test = tst_data[['text', 'labels']]
#
#
#     if LANGUAGE_FINETUNE:
#         train_list = train['text'].tolist()
#         test_list = test['text'].tolist()
#         complete_list = train_list + test_list
#         lm_train = complete_list[0: int(len(complete_list) * 0.8)]
#         lm_test = complete_list[-int(len(complete_list) * 0.2):]
#
#         with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
#             for item in lm_train:
#                 f.write("%s\n" % item)
#
#         with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
#             for item in lm_test:
#                 f.write("%s\n" % item)
#
#         model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
#         model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"),
#                           eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
#         MODEL_NAME = language_modeling_args["best_model_dir"]
#
#     # Train the model
#     print("Started Training")
#
#     train['labels'] = encode(train["labels"])
#     test['labels'] = encode(test["labels"])
#
#     test_sentences = test['text'].tolist()
#     test_preds = np.zeros((len(test), args["n_fold"]))
#
#     MODEL_NAME = arguments.model_name
#     MODEL_TYPE = arguments.model_type
#     cuda_device = int(arguments.cuda_device)
#
#     if args["evaluate_during_training"]:
#         for i in range(args["n_fold"]):
#             if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
#                 shutil.rmtree(args['output_dir'])
#             torch.cuda.set_device(cuda_device)
#             model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
#                                         use_cuda=torch.cuda.is_available(),
#                                         cuda_device=cuda_device)  # You can set class weights by using the optional weight argument
#             train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
#             model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
#                               accuracy=sklearn.metrics.accuracy_score)
#
#             predictions, raw_outputs = model.predict(test_sentences)
#             print(predictions, raw_outputs)
#             test_preds[:, i] = predictions
#             print("Completed Fold {}".format(i))
#             # test['predictions'] = predictions
#         # select majority class of each instance (row)
#         final_predictions = []
#         for row in test_preds:
#             row = row.tolist()
#             final_predictions.append(int(max(set(row), key=row.count)))
#         test['predictions'] = final_predictions
#     else:
#         model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
#                                     use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
#         model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
#         predictions, raw_outputs = model.predict(test_sentences)
#         print(raw_outputs)
#         confidence_df = pd.DataFrame(raw_outputs)
#         test['preds'] = predictions
#         predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
#         predictions_df.to_csv('prediction.csv')
#         confidence_df.to_csv('confidence_result.csv')
#
#         test['predictions'] = predictions
#
#     model.save_model()
#
#     test['predictions'] = decode(test['predictions'])
#     # test['labels'] = decode(test['labels'])
#
#     # time.sleep(5)
#
  # print_information(test, "predictions", "labels")
#     test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
#     # time.sleep(5)
#     # print_information_multi_class(test, "predictions", "labels")
#     #
#     # test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
#     # new = 'python sinhala_deepoffense.py --model_name=arguments.model_name, --model_type=arguments.model_type, --cuda_device=arguments.cuda_device,--train=arguments.train'
#     # exec(new)
#     # import subprocess
#
#     # run your program and collect the string output
#     # cmd = "python ../../sinhala_deepoffense.py --model_name=arguments.model_name, --model_type=arguments.model_type, --cuda_device=arguments.cuda_device,--train=arguments.train"
#     # out_str = subprocess.call(cmd, shell=True)
#     # cm1 = "../../"
#     # # arguments.train = 'data/new_sold.tsv'
#     # out_str1 = subprocess.call(cm1, shell=True)
#     # cmd = "python -m experiments.level_a.sinhala_deepoffense --model_name="+arguments.model_name+" --model_type="+arguments.model_type+" --cuda_device="+arguments.cuda_device+" --train="+arguments.train+""
#     # out_str = subprocess.call(cmd, shell=True)
#
# else:
#     # data = data.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
#     # train = data[['text', 'labels']]
#     #
#     # test_data = pd.read_csv(arguments.test, sep="\t")
#     # test_data = test_data.rename(columns={'tweet': 'text', 'subtask_a': 'labels'})
#     # test =test_data[['text', 'labels']]

trn_data = pd.read_csv(arguments.train, sep="\t")
tst_data = pd.read_csv(arguments.test, sep="\t")

if arguments.lang == "en":
    trn_data, tst_data = train_test_split(trn_data, test_size=0.1)

elif arguments.lang == "sin":
    trn_data = trn_data.rename(columns={'content': 'text', 'Class': 'labels'})
    # tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})

elif arguments.lang == "hin":
    trn_data = trn_data.rename(columns={'task_1': 'labels'})
    tst_data = tst_data.rename(columns={'subtask_a': 'labels', 'tweet': 'text'})

# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text']]
# test = tst_data[['text','labels']]

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

for i in range(args["n_fold"]):
    if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
        shutil.rmtree(args['output_dir'])
    print("Started Fold {}".format(i))
    torch.cuda.set_device(cuda_device)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)  # You can set class weights by using the optional weight argument
    # train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    # model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
    #                   accuracy=sklearn.metrics.accuracy_score)
    # model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
    #                             use_cuda=torch.cuda.is_available())

    predictions, raw_outputs = model.predict(test_sentences)
    probs = softmax(raw_outputs, axis=1)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))
# select majority class of each instance (row)

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))
# test['predictions'] = final_predictions
# else:
# model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
#                             use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)
# # model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
# predictions, raw_outputs = model.predict(test_sentences)

confidence_df = pd.DataFrame(probs)
test['preds'] = predictions
predictions_df = pd.merge(test, test[['preds']], how='left', left_index=True, right_index=True)
predictions_df.to_csv('prediction.csv')
confidence_df.to_csv('confidence_result1.csv', index=False)
test['predictions'] = predictions
df1 = pd.read_csv('prediction.csv')
column_names = ['1', '2']
df = pd.read_csv('confidence_result1.csv', names=column_names, header=None)
frames = [df, df1]
result = pd.concat([df1, df], axis=1)
result.to_csv('one_prediction.csv')

new = []
new1 = []
new2 = []

m1 = np.mean(df['1'])
m2 = np.mean(df['2'])

print(m1,m2)

l1 = np.std(df['1'])
l2 = np.std(df['2'])

# print(l1,l2)

# 0.01, 0.015, 0.02
#
# st1 = l1 / 0.4
# st2 = l2 / 0.4

# print(st1, st2)

# get all the offensive and not offensive posts from the dataset
# df_group_posts = result.groupby('')
# offensive_posts = df_group_posts.get_group('')
# for ix in offensive_posts.index:
#     off_prob = offensive_posts.loc[ix]['']
#     if ((m1 + l1 > off_prob) and (m1 - l1 < off_prob)):
#         new.append(offensive_posts.loc[ix][''])
#
# offensive_not_posts = df_group_posts.get_group('')
# for ix in offensive_not_posts.index:
#     not_off_prob = offensive_not_posts.loc[ix]['']
#     if ((m1 + l1 > not_off_prob) and (m1 - l1 < not_off_prob)):
#         new.append(offensive_not_posts.loc[ix][''])


for ix in df.index:
    e = df.loc[ix]['1']
    # full = e - m1
    f = df.loc[ix]['2']
    # full2 = f - m2

    if ((m1 + l1 > e) and (m1 - l1 < e)):
        new.append(df.loc[ix]['1'])
    else:
        if (m2 + l1 > f):
            new2.append(f.loc[ix]['2'])
    #     # print(new)
    # if (full2 < st2):
    #     new1.append(df.loc[ix]['2'])
        # print(new1)
    # if (full3 > st3):
    #     new2.append(df.loc[ix]['3'])
    #     # print(new2)

# print(l1)
# print(l2)
# print(l3)
#
# print(m1)
# print(m2)
# print(m3)

df_new = result.iloc[np.where(result['1'].isin(new))]
df_new2 = result.iloc[np.where(result['2'].isin(new2))]
# df_new3 = result.iloc[np.where(result['3'].isin(new2))]
# new_dataframe = pd.concat([df_new, df_new2]).drop_duplicates()
new_dataframe = pd.concat([df_new,df_new2]).drop_duplicates()
new_dataframe = df_new.filter(['id', 'text', 'preds_y'])
new_dataframe['preds_y'] = new_dataframe['preds_y'].map({0.0: 'NOT', 1.0: 'OFF'})
new_dataframe.rename({'text': 'content', 'preds_y': 'Class'}, axis=1, inplace=True)
new_dataframe.to_csv('new_train.csv')

model.save_model()

# test['predictions'] = decode(test['predictions'])

# time.sleep(5)
# print_information(test, "predictions", "labels")

test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')

df_nw = pd.read_csv(arguments.train, sep="\t")
df_merged = df_nw.append(new_dataframe, ignore_index=True)
# how to replace this to same argument?????
df_merged.to_csv('data/new_sold.tsv', sep="\t")



