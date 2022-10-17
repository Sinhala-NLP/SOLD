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

params['data_file'] = '/content/SOLD/data/SOLD_test.tsv'
params['class_names'] = '/content/SOLD/deepoffense/explainability/classes_two_SOLD.npy'
params['model_name'] = "sinhala-nlp/sinbert-sold-si"
params['best_params']=False

#TODO: pass by args
model = ExplainableModel("auto", "sinhala-nlp/sinbert-sold-si", args=args,
                                use_cuda=torch.cuda.is_available())

final_list_dict = model.get_final_dict_with_rational(params, params['data_file'], topk=5)

path_name = model_dict_params[model_to_use]

path_name_explanation = 'explanations_dicts/' + path_name.split('/')[1].split('.')[0] + '_' + str(
    params['att_lambda']) + '_explanation_top5.json'
with open(path_name_explanation, 'w') as fp:
    fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))
