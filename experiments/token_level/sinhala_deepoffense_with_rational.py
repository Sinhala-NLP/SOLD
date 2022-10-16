import argparse
import os
import shutil
import json 
import ast
import random

import numpy as np
import pandas as pd
import sklearn
import torch

from deepoffense.explainability import ExplainableModel
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from deepoffense.util.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def return_params(path_name,att_lambda,num_classes=2):
    with open(path_name,mode='r') as f:
        params = json.load(f)
    for key in params:
        if params[key] == 'True':
             params[key]=True
        elif params[key] == 'False':
             params[key]=False
        if( key in ['batch_size','num_classes','hidden_size','supervised_layer_pos','num_supervised_heads','random_seed','max_length']):
            if(params[key]!='N/A'):
                params[key]=int(params[key])

        if((key == 'weights') and (params['auto_weights']==False)):
            params[key] = ast.literal_eval(params[key])
    params['att_lambda']=att_lambda
    params['num_classes']=num_classes
    if(params['bert_tokens']):        
        output_dir = 'Saved/'+params['path_files']+'_'
        if(params['train_att']):
            if(params['att_lambda']>=1):
                params['att_lambda']=int(params['att_lambda'])
            output_dir=output_dir+str(params['supervised_layer_pos'])+'_'+str(params['num_supervised_heads'])
            output_dir=output_dir+'_'+str(params['num_classes'])+'_'+str(params['att_lambda'])

        else:
            output_dir=output_dir+'_'+str(params['num_classes'])
        params['path_files']=output_dir
    
    if(params['num_classes']==2 and (params['auto_weights']==False)):
          params['weights']=[1.0,1.0]
    
    return params

args = {}
args['model_to_use'] = 'bert'
args['attention_lambda'] = 100

model_to_use = args['model_to_use']

model_dict_params = {
    'bert': '/content/SOLD/deepoffense/explainability/bestModel_bert_base_uncased_Attn_train_FALSE.json',
}

params = return_params(model_dict_params[model_to_use], float(args['attention_lambda']))

params['variance'] = 1
params['num_classes'] = 2
params['device'] = 'cuda'

fix_the_random(seed_val=params['random_seed'])

params['data_file'] = '/content/SOLD/data/SOLD_test_annotated.tsv'
params['class_names'] = '/content/SOLD/deepoffense/explainability/classes_two_SOLD.npy'
params['model_name'] = "sinhala-nlp/sinbert-sold-si"
params['best_params']=False

#TODO: pass by args
eval_df = pd.read_csv('/content/SOLD/data/SOLD_test.tsv', sep="\t")

eval_df = eval_df.rename(columns={'content': 'text', 'Class': 'labels'})
eval_df = eval_df[['text', 'labels']]
eval_df['labels'] = encode(eval_df["labels"])

model = ExplainableModel("auto", "sinhala-nlp/sinbert-sold-si", args=args,
                                use_cuda=torch.cuda.is_available())

final_list_dict = model.get_final_dict_with_rational(params, params['data_file'], topk=5)

path_name = model_dict_params[model_to_use]

path_name_explanation = 'explanations_dicts/' + path_name.split('/')[1].split('.')[0] + '_' + str(
    params['att_lambda']) + '_explanation_top5.json'
with open(path_name_explanation, 'w') as fp:
    fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))
    
# results,_,_ = model.eval_model(eval_df)

# data = pd.read_csv('/content/SOLD/data/SOLD_test.tsv', sep="\t")
# data = data.rename(columns={'tweet': 'text', 'subtask_a': 'labels', 'id': 'post_id'})
# data = data[['text', 'labels', 'post_id']]

# train, test = train_test_split(data[:100], test_size=0.2)

# # Train the model
# print("Started Training")

# train['labels'] = encode(train["labels"])
# test['labels'] = encode(test["labels"])

# test_sentences = test['text'].tolist()
# test_preds = np.zeros((len(test), args["n_fold"]))

# if args["evaluate_during_training"]:
#     for i in range(args["n_fold"]):
#         if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
#             shutil.rmtree(args['output_dir'])
#         print("Started Fold {}".format(i))
#         model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
#                                     use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
#         train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
#         model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
#         predictions, raw_outputs = model.predict(test_sentences)
#         test_preds[:, i] = predictions
#         print("Completed Fold {}".format(i))
#     # select majority class of each instance (row)
#     final_predictions = []
#     for row in test_preds:
#         row = row.tolist()
#         final_predictions.append(int(max(set(row), key=row.count)))
#     test['predictions'] = final_predictions
# else:
#     model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=3,
#                                 use_cuda=torch.cuda.is_available())
#     model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
#     predictions, raw_outputs = model.predict(test_sentences)
#     test['predictions'] = predictions

# test['predictions'] = decode(test['predictions'])
# test['labels'] = decode(test['labels'])

# time.sleep(5)

# print_information_multi_class(test, "predictions", "labels")
# test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')

# #TODO: Refactor above code
# #TODO: Parameterize (1) train and test with rationals (2) test with rationals
# gc.collect()

# #TODO: Add separate condition to load and rational only
# # model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args, use_cuda=torch.cuda.is_available())

# list_dict, test_data = model.standaloneEval_with_rational(test_sentences, test_data = test)