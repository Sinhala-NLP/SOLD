from tqdm.auto import tqdm, trange
import numpy as np
from numpy import array, exp
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
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

#### Few helper functions to convert attention vectors in 0 to 1 scale. While softmax converts all the values such that their sum lies between 0 --> 1. Sigmoid converts each value in the vector in the range 0 -> 1.
def encodeData(dataframe, vocab, params):
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if (params['bert_tokens']):
            tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
        else:
            list_token_id = []
            for word in row['Text']:
                try:
                    index = vocab.stoi[word]
                except KeyError:
                    index = vocab.stoi['unk']
                list_token_id.append(index)
            tuple_new_data.append((list_token_id, row['Attention'], row['Label']))
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
    data_all_labelled.tokens = data_all_labelled.tokens.str.split()
    data_all_labelled.rationales = data_all_labelled.rationales.apply(lambda x: ast.literal_eval(x))
    data_all_labelled['final_label'] = data_all_labelled['label']
    train_data = get_training_data(data_all_labelled, params, tokenizer)
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
        )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
            ' ',  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,

        )
        ### decide what to later

    return encoded_sent


def ek_extra_preprocess(text, params, tokenizer):
    remove_words = ['<allcaps>', '</allcaps>', '<hashtag>', '</hashtag>', '<elongated>', '<emphasis>', '<repeated>',
                    '\'', 's']
    word_list = text_processor.pre_process_doc(text)
    if (params['include_special']):
        pass
    else:
        word_list = list(filter(lambda a: a not in remove_words, word_list))
    if (params['bert_tokens']):
        sent = " ".join(word_list)
        sent = re.sub(r"[<\*>]", " ", sent)
        sub_word_list = custom_tokenize(sent, tokenizer)
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
    text_tokens = row['tokens']

    ##### a very rare corner case
    if (len(text_tokens) == 0):
        text_tokens = ['dummy']
        print("length of text ==0")
    #####

    mask_all = row['rationales']
    mask_all_temp = mask_all
    count_temp = 0
    if len(mask_all_temp) == 0:
        mask_all_temp.append([0] * len(text_tokens))
    else:
        mask_all_temp = [mask_all]

    word_mask_all = []
    word_tokens_all = []

    for mask in mask_all_temp:
        if (mask[0] == -1):
            mask = [0] * len(mask)

        list_pos = []
        mask_pos = []

        flag = 0
        for i in range(0, len(mask)):
            if (i == 0 and mask[i] == 0):
                list_pos.append(0)
                mask_pos.append(0)

            if (flag == 0 and mask[i] == 1):
                mask_pos.append(1)
                list_pos.append(i)
                flag = 1

            elif (flag == 1 and mask[i] == 0):
                flag = 0
                mask_pos.append(0)
                list_pos.append(i)
        if (list_pos[-1] != len(mask)):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts = []
        for i in range(len(list_pos) - 1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i + 1]])

        if (params['bert_tokens']):
            word_tokens = [101]
            word_mask = [0]
        else:
            word_tokens = []
            word_mask = []

        for i in range(0, len(string_parts)):
            tokens = ek_extra_preprocess(" ".join(string_parts[i]), params, tokenizer)
            masks = [mask_pos[i]] * len(tokens)
            word_tokens += tokens
            word_mask += masks

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
    if (row['final_label'] in ['normal', 'non-toxic']):
        at_mask_fin = [1 / len(at_mask[0]) for x in at_mask[0]]
    else:
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


def get_test_data(data, params, tokenizer = None, message='text'):
    '''input: data is a dataframe text ids labels column only'''
    '''output: training data in the columns post_id,text (tokens) , attentions (normal) and labels'''
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    raw_text_list = []
    rationale_list = []
    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        post_id = row['post_id']
        annotation = row['final_label']
        text = row['text']
        rationales = row['rationales']
        tokens_all, attention_masks = returnMask(row, params, tokenizer)
        attention_vector = aggregate_attention(attention_masks, row, params)
        attention_list.append(attention_vector)
        text_list.append(tokens_all)
        label_list.append(annotation)
        post_ids_list.append(post_id)
        raw_text_list.append(text)
        rationale_list.append(rationales)

    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list, raw_text_list, rationale_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label', 'Raw Text List', 'Rationales'])
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
        annotation = row['final_label']

        text = row['text']
        post_id = row['post_id']
        annotation_list = [row['label']]  # ,row['label2'],row['label3']]
        tokens_all = list(row['text'])
        #         attention_masks =  [list(row['explain1']),list(row['explain2']),list(row['explain1'])]

        if (annotation != 'undecided'):
            tokens_all, attention_masks = returnMask(row, params, tokenizer)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output


def get_training_data(data, params, tokenizer):
    '''input: data is a dataframe text ids attentions labels column only'''
    '''output: training data in the columns post_id,text, attention and labels '''

    majority = params['majority']
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    count = 0
    count_confused = 0
    raw_text_list = []
    rationale_list = []
    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # print(row)
        # print(params)
        text = row['text']
        post_id = row['post_id']

        # annotation_list=[row['label'],row['label2'],row['label3']]
        annotation = row['label']
        rationales = row['rationales']

        if (annotation != 'undecided'):
            tokens_all, attention_masks = returnMask(row, params, tokenizer)
            attention_vector = aggregate_attention(attention_masks, row, params)
            attention_list.append(attention_vector)
            text_list.append(tokens_all)
            label_list.append(annotation)
            post_ids_list.append(post_id)
            raw_text_list.append(text)
            rationale_list.append(rationales)
        else:
            count_confused += 1

    print("attention_error:", count)
    print("no_majority:", count_confused)
    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list, raw_text_list, rationale_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label', 'Raw Text List', 'Rationales'])

    filename = set_name(params)
    training_data.to_pickle(filename)
    return training_data


# TODO: remove else of params['bert_tokens'] checks

def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end=False,
        sep_token_extra=False,
        pad_on_left=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        multi_label=False,
        silent=False,
        use_multiprocessing=True,
        sliding_window=False,
        flatten=False,
        stride=None,
        add_prefix_space=False,
        pad_to_max_length=True,
        args=None,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            multi_label,
            stride,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]

    if use_multiprocessing:
        if args.multiprocessing_chunksize == -1:
            chunksize = max(len(examples) // (args.process_count * 2), 500)
        else:
            chunksize = args.multiprocessing_chunksize
        if sliding_window:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature_sliding_window,
                            examples,
                            chunksize=chunksize,
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            with Pool(process_count) as p:
                features = list(
                    tqdm(
                        p.imap(
                            convert_example_to_feature, examples, chunksize=chunksize
                        ),
                        total=len(examples),
                        disable=silent,
                    )
                )
    else:
        if sliding_window:
            features = [
                convert_example_to_feature_sliding_window(example)
                for example in tqdm(examples, disable=silent)
            ]
            if flatten:
                features = [
                    feature for feature_set in features for feature in feature_set
                ]
        else:
            features = [
                convert_example_to_feature(example)
                for example in tqdm(examples, disable=silent)
            ]

    return features


def convert_example_to_feature(
        example_row,
        params,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    bboxes = []
    if example.bboxes:
        tokens_a = []
        for word, bbox in zip(example.text_a.split(), example.bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens_a.extend(word_tokens)
            bboxes.extend([bbox] * len(word_tokens))

        cls_token_box = [0, 0, 0, 0]
        sep_token_box = [1000, 1000, 1000, 1000]
        pad_token_box = [0, 0, 0, 0]

    else:
        if add_prefix_space and not example.text_a.startswith(" "):
            tokens_a = tokenizer.tokenize(" " + example.text_a)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        if add_prefix_space and not example.text_b.startswith(" "):
            tokens_b = tokenizer.tokenize(" " + example.text_b)
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]
            if example.bboxes:
                bboxes = bboxes[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if bboxes:
        bboxes += [sep_token_box]

    if tokens_b:
        if sep_token_extra:
            tokens += [sep_token]
            segment_ids += [sequence_b_segment_id]

        tokens += tokens_b + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        if bboxes:
            bboxes = [cls_token_box] + bboxes

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    tokens_all, attention_mask = returnMask(example, params, tokenizer)

    # Zero-pad up to the sequence length.
    if pad_to_max_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                                 [0 if mask_padding_with_zero else 1] * padding_length
                         ) + input_mask
            attention_mask = (
                                     [0 if mask_padding_with_zero else 1] * padding_length
                             ) + attention_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
            )
            attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if bboxes:
                bboxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if bboxes:
            assert len(bboxes) == max_seq_length
    # if output_mode == "classification":
    #     label_id = label_map[example.label]
    # elif output_mode == "regression":
    #     label_id = float(example.label)
    # else:
    #     raise KeyError(output_mode)

    # if output_mode == "regression":
    #     label_id = float(example.label)

    if bboxes:
        return InputFeaturesWithRationales(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=example.label,
            attention_mask=attention_mask,
            bboxes=bboxes,
        )
    else:
        return InputFeaturesWithRationales(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            label_id=example.label,
        )


def convert_example_to_feature_sliding_window(
        example_row,
        params,
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        sep_token_extra=False,
):
    (
        example,
        max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end,
        cls_token,
        sep_token,
        cls_token_segment_id,
        pad_on_left,
        pad_token_segment_id,
        sep_token_extra,
        multi_label,
        stride,
        pad_token,
        add_prefix_space,
        pad_to_max_length,
    ) = example_row

    if stride < 1:
        stride = int(max_seq_length * stride)

    bucket_size = max_seq_length - (3 if sep_token_extra else 2)
    token_sets = []

    if add_prefix_space and not example.text_a.startswith(" "):
        tokens_a = tokenizer.tokenize(" " + example.text_a)
    else:
        tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > bucket_size:
        token_sets = [
            tokens_a[i: i + bucket_size] for i in range(0, len(tokens_a), stride)
        ]
    else:
        token_sets.append(tokens_a)

    if example.text_b:
        raise ValueError(
            "Sequence pair tasks not implemented for sliding window tokenization."
        )

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    input_features = []
    for tokens_a in token_sets:
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                                 [0 if mask_padding_with_zero else 1] * padding_length
                         ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)

        input_features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=example.label,
            )
        )

    return input_features


class InputFeaturesWithRationales(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, attention_mask, bboxes=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.attention_mask = attention_mask
        if bboxes:
            self.bboxes = bboxes


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
        test_data_modified.append([row['Post_id'], new_text, new_attention, row['Label'], row['Raw Text List'], row['Rationales']])

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

    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id,
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

        if majority_label == 'normal':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

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