from tqdm.auto import tqdm, trange
import numpy as np
from numpy import array, exp
import pandas as pd
from multiprocessing import Pool, cpu_count
import datetime
from os import path
import pickle
from gensim.models import KeyedVectors
from transformers.models.auto.tokenization_auto import AutoTokenizer
import ast
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import string
import re
import os
import more_itertools as mit
import json
from transformers.models.bert.modeling_bert import *
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)

from transformers import BertPreTrainedModel


def cross_entropy(input1, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))


def masked_cross_entropy(input1, target, mask):
    cr_ent = 0
    for h in range(0, mask.shape[0]):
        cr_ent += cross_entropy(input1[h][mask[h]], target[h][mask[h]])

    return cr_ent / mask.shape[0]

#### Few helper functions to convert attention vectors in 0 to 1 scale. While softmax converts all the values such that their sum lies between 0 --> 1. Sigmoid converts each value in the vector in the range 0 -> 1.
def encodeData(dataframe, vocab, params):
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if (params['bert_tokens']):
            tuple_new_data.append((row['Text'], row['Attention'], row['Label'], row['Raw_text']))
        else:
            list_token_id = []
            for word in row['Text']:
                try:
                    index = vocab.stoi[word]
                except KeyError:
                    index = vocab.stoi['unk']
                list_token_id.append(index)
            tuple_new_data.append((list_token_id, row['Attention'], row['Label'], row['Raw_text']))
    return tuple_new_data


def set_name(params):
    file_name = 'data/Total_data'
    if (params['bert_tokens']):
        file_name += '_bert'
    else:
        file_name += '_normal'

    file_name += '_' + params['type_attention'] + '_' + str(params['variance']) + '_' + str(params['max_length'])
    if (params['decay']):
        file_name += '_' + params['method'] + '_' + str(params['window']) + '_' + str(params['alpha']) + '_' + str(
            params['p_value'])
    file_name += '_' + str(params['num_classes']) + '.pickle'
    return file_name


def collect_data(params):
    if (params['bert_tokens']):
        print('Loading BERT tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    else:
        tokenizer = None
    # TODO: Add a fork here
    # data_all_labelled=get_annotated_data(params)
    data_all_labelled = pd.read_csv(params['data_file'], sep="\t")  # , nrows=10)
    data_all_labelled['raw_text'] = data_all_labelled['text']
    data_all_labelled.text = data_all_labelled.tokens.str.split()
    data_all_labelled.rationales = data_all_labelled.rationales.apply(lambda x: [ast.literal_eval(x)])
    data_all_labelled['final_label'] = data_all_labelled['label']
    # print('test')
    # print(data_all_labelled.iloc[4]['rationales'])
    train_data = get_test_data(data_all_labelled, params, tokenizer,
                               message='text')  # get_training_data(data_all_labelled, params, tokenizer)
    return train_data


def createDatasetSplit(params):
    filename = set_name(params)
    if path.exists(filename):
        ##### REMOVE LATER ######
        # dataset=collect_data(params)
        pass
    else:
        params['data_file'] = '/content/SOLD/data/SOLD_test.tsv'

    if (path.exists(filename[:-7])):
        with open(filename[:-7] + '/test_data.pickle', 'rb') as f:
            X_test = pickle.load(f)
        if (params['bert_tokens'] == False):
            with open(filename[:-7] + '/vocab_own.pickle', 'rb') as f:
                vocab_own = pickle.load(f)


    else:
        if (params['bert_tokens'] == False):
            word2vecmodel1 = KeyedVectors.load("Data/word2vec.model")
            vector = word2vecmodel1['easy']
            assert (len(vector) == 300)

        X_test = collect_data(params)

        # TODO: Add datafiles later
        if (params['bert_tokens']):
            vocab_own = None
            vocab_size = 0
            padding_idx = 0
        else:
            vocab_own = Vocab_own(X_train, word2vecmodel1)
            vocab_own.create_vocab()
            padding_idx = vocab_own.stoi['<pad>']
            vocab_size = len(vocab_own.vocab)

        X_test = encodeData(X_test, vocab_own, params)

        print("total dataset size:", len(X_test))

        os.mkdir(filename[:-7])
        with open(filename[:-7] + '/test_data.pickle', 'wb') as f:
            pickle.dump(X_test, f)
        if (params['bert_tokens'] == False):
            with open(filename[:-7] + '/vocab_own.pickle', 'wb') as f:
                pickle.dump(vocab_own, f)

    if (params['bert_tokens'] == False):
        return X_test, vocab_own
    else:
        return X_test




text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    # corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def custom_tokenize(sent, tokenizer, max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    try:

        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # max_length = max_length,
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
            # max_length=max_length,
            # return_tensors='pt'
        )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
            ' ',  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,
            # return_tensors='pt'
        )
        ### decide what to later

    return encoded_sent

def ek_extra_preprocess(text, params, tokenizer):
    # remove_words = ['<allcaps>', '</allcaps>', '<hashtag>', '</hashtag>', '<elongated>', '<emphasis>', '<repeated>',
    #                 '\'', 's']
    # word_list = text_processor.pre_process_doc(text)
    # if (params['include_special']):
    #     pass
    # else:
    #     word_list = list(filter(lambda a: a not in remove_words, word_list))
    word_list = list(text)
    # print(type(word_list))
    if (params['bert_tokens']):
        # sent = " ".join(word_list)
        # sent = re.sub(r"[<\*>]", " ", sent)
        sub_word_list = custom_tokenize(text, tokenizer)
        # print('bert tokens')
        # sub_word_list = tokenizer.tokenize(text)
        return sub_word_list
    else:
        word_list = [token for token in word_list if token not in string.punctuation]
        return word_list

##### We mostly use softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Here we convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)

def sigmoid(z):
    """Compute sigmoid values"""
    g = 1 / (1 + exp(-z))
    return g

def returnMask(row, params, tokenizer):
    # text_tokens = row['text']
    text_tokens = row['text']

    ##### a very rare corn=r case
    if (len(text_tokens) == 0):
        text_tokens = ['dummy']
        print("length of text ==0")
    #####
    mask_all = row['rationales']
    mask_all_temp = mask_all
    if len(mask_all[0]) == 0:
        mask_all_temp = []


    count_temp = 0
    # while (len(mask_all_temp) != 1):
        # mask_all_temp.append([0] * len(text_tokens))
    while (len(mask_all_temp) != 3):
        mask_all_temp.append([0] * len(text_tokens))

    word_mask_all = []
    word_tokens_all = []

    # TODO: Here we consider rationales one by one instead of string parts in HateXplain
    for mask in mask_all_temp:
        if (params['bert_tokens']):
            word_tokens = [101]
            word_mask = [0]
        else:
            word_tokens = []
            word_mask = []

        for i in range(0, len(mask)):
            # TODO: remove join
            tokens = custom_tokenize(text_tokens[i], tokenizer)
            masks = [mask[i]] * len(tokens)
            # masks = custom_tokenize(" ".join(mask[i]), tokenizer)
            word_tokens += tokens
            word_mask += masks

        # if (mask[0] == -1):
        #     mask = [0] * len(mask)
        #
        # list_pos = []
        # mask_pos = []
        #
        # flag = 0
        # for i in range(0, len(mask)):
        #     if (i == 0 and mask[i] == 0):
        #         list_pos.append(0)
        #         mask_pos.append(0)
        #
        #     if (flag == 0 and mask[i] == 1):
        #         mask_pos.append(1)
        #         list_pos.append(i)
        #         flag = 1
        #
        #     elif (flag == 1 and mask[i] == 0):
        #         flag = 0
        #         mask_pos.append(0)
        #         list_pos.append(i)
        # if (list_pos[-1] != len(mask)):
        #     list_pos.append(len(mask))
        #     mask_pos.append(0)
        # string_parts = []
        # for i in range(len(list_pos) - 1):
        #     string_parts.append(text_tokens[list_pos[i]:list_pos[i + 1]])
        #
        # if (params['bert_tokens']):
        #     word_tokens = [101]
        #     word_mask = [0]
        # else:
        #     word_tokens = []
        #     word_mask = []
        #
        # for i in range(0, len(string_parts)):
        #     tokens = custom_tokenize(" ".join(string_parts[i]), tokenizer)
        #     masks = [mask_pos[i]] * len(tokens)
        #     word_tokens += tokens
        #     word_mask += masks
        #     #
        #     # if row['post_id'] == 1305480579375198208:
        #     #     print('custom_tokenize[i]')
        #     #     print(custom_tokenize(" ".join(string_parts[i]), tokenizer))
        #     #     print('tok')
        #     #     print(tokenizer.encode(" ".join(string_parts[i])))
        #     #     print('tokens')
        #     #     print(tokens)
        #     #     print('masks')
        #     #     print(masks)
        #
        if (params['bert_tokens']):
            ### always post truncation
            word_tokens = word_tokens[0:(int(params['max_length']) - 2)]
            word_mask = word_mask[0:(int(params['max_length']) - 2)]
            word_tokens.append(102)
            word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)

        #     for k in range(0,len(mask_all)):
        #          if(mask_all[k][0]==-1):
        #             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if (len(mask_all) == 0):
        word_mask_all = []
    else:
        word_mask_all = word_mask_all[0:len(mask_all)]
    return word_tokens_all[0], word_mask_all

def aggregate_attention(at_mask, row, params):
    """input: attention vectors from 2/3 annotators (at_mask), row(dataframe row), params(parameters_dict)
           function: aggregate attention from different annotators.
           output: aggregated attention vector"""

    #### If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    if (row['final_label'] in ['NOT']):
        # print('NOT')
        at_mask_fin = [1 / len(at_mask[0]) for x in at_mask[0]]
    else:
        # print('OFF')
        at_mask_fin = at_mask
        #### Else it will choose one of the options, where variance is added, mean is calculated, finally the vector is normalised.
        if (params['type_attention'] == 'sigmoid'):
            at_mask_fin = int(params['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = sigmoid(at_mask_fin)
        elif (params['type_attention'] == 'softmax'):
            at_mask_fin = int(params['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = softmax(at_mask_fin)
        elif (params['type_attention'] == 'neg_softmax'):
            at_mask_fin = int(params['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = neg_softmax(at_mask_fin)
        elif (params['type_attention'] in ['raw', 'individual']):
            pass
    if (params['decay'] == True):
        at_mask_fin = decay(at_mask_fin, params)

    return at_mask_fin

##### Decay and distribution functions.To decay the attentions left and right of the attented word. This is done to decentralise the attention to a single word.
def distribute(old_distribution, new_distribution, index, left, right, params):
    window = params['window']
    alpha = params['alpha']
    p_value = params['p_value']
    method = params['method']

    reserve = alpha * old_distribution[index]
    #     old_distribution[index] = old_distribution[index] - reserve

    if method == 'additive':
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

        for temp in range(index + 1, index + right):
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

    if method == 'geometric':
        # we first generate the geometric distributio for the left side
        temp_sum = 0.0
        newprob = []
        for temp in range(left):
            each_prob = p_value * ((1.0 - p_value) ** temp)
            newprob.append(each_prob)
            temp_sum += each_prob
            newprob = [each / temp_sum for each in newprob]

        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve * newprob[-(temp - (index - left)) - 1]

        # do the same thing for right, but now the order is opposite
        temp_sum = 0.0
        newprob = []
        for temp in range(right):
            each_prob = p_value * ((1.0 - p_value) ** temp)
            newprob.append(each_prob)
            temp_sum += each_prob
            newprob = [each / temp_sum for each in newprob]
        for temp in range(index + 1, index + right):
            new_distribution[temp] = new_distribution[temp] + reserve * newprob[temp - (index + 1)]

    return new_distribution

def decay(old_distribution, params):
    window = params['window']
    new_distribution = [0.0] * len(old_distribution)
    for index in range(len(old_distribution)):
        right = min(window, len(old_distribution) - index)
        left = min(window, index)
        new_distribution = distribute(old_distribution, new_distribution, index, left, right, params)

    if (params['normalized']):
        norm_distribution = []
        for index in range(len(old_distribution)):
            norm_distribution.append(old_distribution[index] + new_distribution[index])
        tempsum = sum(norm_distribution)
        new_distrbution = [each / tempsum for each in norm_distribution]
    return new_distribution

def get_test_data(data, params, tokenizer=None, message='text'):
    '''input: data is a dataframe text ids labels column only'''
    '''output: training data in the columns post_id,text (tokens) , attentions (normal) and labels'''
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    raw_text_list = []

    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        post_id = row['post_id']
        annotation = row['final_label']
        raw_text = row['raw_text']
        tokens_all, attention_masks = returnMask(row, params, tokenizer)
        attention_vector = aggregate_attention(attention_masks, row, params)
        attention_list.append(attention_vector)
        text_list.append(tokens_all)
        label_list.append(annotation)
        post_ids_list.append(post_id)
        raw_text_list.append(raw_text)

        # Calling DataFrame constructor after zipping
        # both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list, raw_text_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label', 'Raw_text'])

    return training_data

# Load the whole dataset and get the tokenwise rationales
def get_training_data_eraser(data, params, tokenizer):
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []

    final_binny_output = []
    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # if index ==0:
        #     print(row)
        annotation = row['final_label']

        # text = row['Raw_text']
        post_id = row['post_id']
        annotation_list = [row['label'], row['label'], row['label']]
        tokens_all = list(row['text'])
        #         attention_masks =  [list(row['explain1']),list(row['explain2']),list(row['explain1'])]

        if (annotation != 'undecided'):
            tokens_all, attention_masks = returnMask(row, params, tokenizer)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    if post_id == 1290554471714496514:
        print(final_binny_output)
    return final_binny_output

# TODO: remove else of params['bert_tokens'] checks
def combine_features(tuple_data, params, is_train=False):
    input_ids = [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele[2] for ele in tuple_data]
    raw_text = [ele[3] for ele in tuple_data]

    encoder = LabelEncoder()

    encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
    labels = encoder.transform(labels)

    input_ids = pad_sequences(input_ids, maxlen=int(params['max_length']), dtype="long",
                              value=0, truncating="post", padding="post")
    att_vals = pad_sequences(att_vals, maxlen=int(params['max_length']), dtype="float",
                             value=0.0, truncating="post", padding="post")
    att_masks = custom_att_masks(input_ids)
    dataloader = return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train)
    return dataloader

def return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(np.array(att_masks), dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals), dtype=torch.float)
    data = TensorDataset(inputs, attention, masks, labels)
    sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader

def custom_att_masks(input_ids):
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks

def convert_data(test_data, params, list_dict, rational_present=True, topk=2):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""

    temp_dict = {}
    for ele in list_dict:
        temp_dict[ele['annotation_id']] = ele['rationales'][0]['soft_rationale_predictions']

    test_data_modified = []

    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            attention = temp_dict[row['Post_id']]
        except KeyError:
            continue
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text = []
        new_attention = []
        if (rational_present):
            if (params['bert_tokens']):
                new_attention = [0]
                new_text = [101]
            for i in range(len(row['Text'])):
                if (i in topk_indices):
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
            if (params['bert_tokens']):
                new_attention.append(0)
                new_text.append(102)
        else:
            for i in range(len(row['Text'])):
                if (i not in topk_indices):
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
        test_data_modified.append([row['Post_id'], new_text, new_attention, row['Label'], row['Raw_text']])

    df = pd.DataFrame(test_data_modified, columns=test_data.columns)
    return df

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    temp = e_x / e_x.sum(axis=0)  # only difference

    if np.isnan(temp).any() == True:
        return [0.0, 1.0, 0.0]
    else:
        return temp

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each == 1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each) == int:
            start = each
            end = each + 1
        elif len(each) == 2:
            start = each[0]
            end = each[1] + 1
        else:
            print('error')

        output.append({"docid": post_id,
                       "end_sentence": -1,
                       "end_token": end,
                       "start_sentence": -1,
                       "start_token": start,
                       "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    final_output = []

    if save_split:
        train_fp = open(save_path + 'train.jsonl', 'w')
        val_fp = open(save_path + 'val.jsonl', 'w')
        test_fp = open(save_path + 'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):

        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if post_id == 1290554471714496514:
            print(eachrow)
        #
        # if majority_label == 'NOT':
        #     continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]
        # final_explanation =explanations[0]# [int(each) for each in explanations[0]]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path + 'docs'):
                os.makedirs(save_path + 'docs')

            with open(save_path + 'docs/' + str(post_id), 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['test']:
                test_fp.write(json.dumps(temp) + '\n')
            else:
                print(post_id)

    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    return final_output

class SC_weighted_BERT(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, weight=None):
        super(SC_weighted_BERT, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.weight = weight
        # TODO: Pass params
        self.num_sv_heads = 1
        self.sv_layer = 11
        self.lam = 100.0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        attention_vals = None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            # attention_vals = attention_vals
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_att=0
                for i in range(self.num_sv_heads):
                    attention_weights = outputs[1][self.sv_layer][:, i, 0, :]
                    loss_att += self.lam * masked_cross_entropy(attention_weights, attention_vals, attention_mask)

                loss = loss + loss_att
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
# class SC_weighted_BERT(BertPreTrainedModel):
#     def __init__(self, config, params):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.weights = params['weights']
#         self.train_att = params['train_att']
#         self.lam = params['att_lambda']
#         self.num_sv_heads = params['num_supervised_heads']
#         self.sv_layer = params['supervised_layer_pos']
#         self.bert = AutoModelForSequenceClassification.from_pretrained(params['model_name'], config=config)#, output_hidden_states=True)#BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.softmax=nn.Softmax(config.num_labels)
#         self.init_weights()
#
#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 attention_vals=None,
#                 token_type_ids=None,
#                 position_ids=None,
#                 head_mask=None,
#                 inputs_embeds=None,
#                 labels=None,
#                 device=None):
#
#         # outputs = self.bert(input_ids,
#         #                        attention_mask=attention_mask,
#         #                        token_type_ids=token_type_ids,
#         #                        position_ids=position_ids,
#         #                        head_mask=head_mask)
#         # logits = outputs[0]
#         # logits = self.classifier(sequence_output)
#
#         # outputs = self.bert(
#         #     input_ids,
#         #     attention_mask=attention_mask,
#         #     token_type_ids=token_type_ids,
#         #     position_ids=position_ids,
#         #     head_mask=head_mask,
#         #     inputs_embeds=inputs_embeds,
#         # )
#
#         # outputs = model(b_input_ids,
#         #                 output_attentions=True,
#         #                 output_hidden_states=False,
#         #                 labels=None)
#         outputs = self.bert(
#             input_ids,
#             output_attentions=True,
#             output_hidden_states=False,
#             # labels=None
#         )
#
#         # pooled_output = outputs[0]
#         # #
#         # pooled_output = self.dropout(pooled_output)
#
#         logits = outputs[0] #self.classifier(pooled_output)
#         # logits = self.softmax(logits)
#         # logits = outputs[0]
#
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#         # print('test2')
#
#         if labels is not None:
#             # print('test3')
#             loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
#             loss_logits = loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
#             loss = loss_logits
#             if (self.train_att):
#
#                 loss_att = 0
#                 for i in range(self.num_sv_heads):
#                     attention_weights = outputs[1][self.sv_layer][:, i, 0, :]
#                     loss_att += self.lam * masked_cross_entropy(attention_weights, attention_vals, attention_mask)
#                 loss = loss + loss_att
#             outputs = (loss,) + outputs
#
#         return outputs  # (loss), logits, (hidden_states), (attentions)


class Vocab_own():
    def __init__(self, dataframe, model):
        self.itos = {}
        self.stoi = {}
        self.vocab = {}
        self.embeddings = []
        self.dataframe = dataframe
        self.model = model

    ### load embedding given a word and unk if word not in vocab
    ### input: word
    ### output: embedding,word or embedding for unk, unk
    def load_embeddings(self, word):
        try:
            return self.model[word], word
        except KeyError:
            return self.model['unk'], 'unk'

    ### create vocab,stoi,itos,embedding_matrix
    ### input: **self
    ### output: updates class members
    def create_vocab(self):
        count = 1
        for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            for word in row['Text']:
                vector, word = self.load_embeddings(word)
                try:
                    self.vocab[word] += 1
                except KeyError:
                    if (word == 'unk'):
                        print(word)
                    self.vocab[word] = 1
                    self.stoi[word] = count
                    self.itos[count] = word
                    self.embeddings.append(vector)
                    count += 1
        self.vocab['<pad>'] = 1
        self.stoi['<pad>'] = 0
        self.itos[0] = '<pad>'
        self.embeddings.append(np.zeros((300,), dtype=float))
        self.embeddings = np.array(self.embeddings)
        # print(self.embeddings.shape)