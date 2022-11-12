import os
import json
import ast
import random

from tqdm.notebook import tqdm
import more_itertools as mit

import numpy as np
import torch
import argparse

from deepoffense.explainability import ExplainableModel
from deepoffense.common.deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE
from deepoffense.explainability.explainable_utils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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


def fix_the_random(seed_val=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def return_params(path_name, att_lambda,model_name,model_to_use, num_classes=2):
    with open(path_name, mode='r') as f:
        params = json.load(f)
    for key in params:
        if params[key] == 'True':
            params[key] = True
        elif params[key] == 'False':
            params[key] = False
        if (key in ['batch_size', 'num_classes', 'hidden_size', 'supervised_layer_pos', 'num_supervised_heads',
                    'random_seed', 'max_length']):
            if (params[key] != 'N/A'):
                params[key] = int(params[key])

        if ((key == 'weights') and (params['auto_weights'] == False)):
            params[key] = ast.literal_eval(params[key])
    params['att_lambda'] = att_lambda
    params['num_classes'] = num_classes
    if (params['bert_tokens']):
        output_dir = 'Saved/' + params['path_files'] + '_'
        if (params['train_att']):
            if (params['att_lambda'] >= 1):
                params['att_lambda'] = int(params['att_lambda'])
            output_dir = output_dir + str(params['supervised_layer_pos']) + '_' + str(params['num_supervised_heads'])
            output_dir = output_dir + '_' + str(params['num_classes']) + '_' + str(params['att_lambda'])

        else:
            output_dir = output_dir + '_' + str(params['num_classes'])
        params['path_files'] = output_dir

    if (params['num_classes'] == 2 and (params['auto_weights'] == False)):
        params['weights'] = [1.0, 1.0]

    params['model_to_use'] = model_to_use
    params['variance'] = 1
    params['device'] = 'cuda'

    params['data_file'] = 'data/SOLD_test.tsv'
    params['class_names'] = 'deepoffense/explainability/classes_two_SOLD.npy'
    params['model_name'] =model_name #"sinhala-nlp/sinbert-sold-si"
    params['best_params'] = False

    #TODO: pass these params from args file or user args
    return params


def generate_explanation_dictionary(params):
    fix_the_random(seed_val=params['random_seed'])

    # TODO: pass by args
    model = ExplainableModel("auto", params['model_name'], args=args,
                             params = params,
                             use_cuda=torch.cuda.is_available())

    final_list_dict = model.get_final_dict_with_rational(params, params['data_file'], topk=5)

    path_name = 'deepoffense/explainability/bestModel_bert_base_uncased_Attn_train_FALSE.json'

    path_name_explanation = 'explanations_dicts/' + path_name.split('/')[1].split('.')[0] + '_' + str(
        params['att_lambda']) + '_explanation_top5.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))

def convert_to_eraser(params):
    data_all_labelled = pd.read_csv(params['data_file'], sep="\t")
    data_all_labelled['raw_text'] = data_all_labelled['text']
    data_all_labelled.text = data_all_labelled.tokens.str.split()
    data_all_labelled.rationales = data_all_labelled.rationales.apply(lambda x: [ast.literal_eval(x)])
    data_all_labelled['final_label'] = data_all_labelled['label']

    # TODO: use tokenizer in explainability model
    if (params['bert_tokens']):
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    else:
        print('Loading Normal tokenizer...')
        tokenizer = None

    training_data=get_training_data_eraser(data_all_labelled, params, tokenizer)

    # The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
    with open('deepoffense/explainability/post_id_divisions.json') as fp:
        id_division = json.load(fp)

    method = 'union'
    save_split = True
    save_path = 'data/Evaluation/Model_Eval/'  # The dataset in Eraser Format will be stored here.
    output_eraser = convert_to_eraser_format(training_data, method, save_split, save_path, id_division)

    return output_eraser

if __name__ == '__main__':
    # model_to_use = 'bert'
    attention_lambda = 100

    model_dict_param = 'deepoffense/explainability/bestModel_bert_base_uncased_Attn_train_FALSE.json'

    parser = argparse.ArgumentParser(
    description='''calculate explanation metrics  ''')
    parser.add_argument('--model_name', required=False, help='model name', default="sinhala-nlp/sinbert-sold-si")
    parser.add_argument('--model_to_use', required=False, help='model type', default="roberta")
    arguments = parser.parse_args()

    model_name = arguments.model_name
    model_to_use = arguments.model_to_use
    params = return_params(model_dict_param, float(attention_lambda), model_name, model_to_use)

    generate_explanation_dictionary(params)
    output_eraser = convert_to_eraser(params)


