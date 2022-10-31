import logging
import os
import random
import warnings

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from deepoffense.explainability.explainable_utils import get_test_data, convert_data, softmax, encodeData, \
    createDatasetSplit, SC_weighted_BERT, combine_features

from deepoffense.classification import ClassificationModel

from tqdm.auto import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer, AlbertConfig, AlbertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    BertTokenizerFast, BertweetTokenizer, CamembertConfig, CamembertTokenizerFast, DebertaConfig,
    DebertaForSequenceClassification, DebertaTokenizer, DistilBertTokenizerFast, ElectraConfig, ElectraTokenizerFast,
    FlaubertConfig, FlaubertTokenizer, LayoutLMConfig, LongformerTokenizerFast, RobertaTokenizerFast,
    XLMRobertaTokenizerFast, XLNetConfig, XLNetTokenizerFast,
)
from transformers.convert_graph_to_onnx import convert, quantize

from deepoffense.classification.classification_utils import sweep_config_to_sweep_values
from deepoffense.classification.transformer_models.albert_model import AlbertForSequenceClassification
from deepoffense.classification.transformer_models.args.model_args import ClassificationArgs
from deepoffense.classification.transformer_models.bert_model import BertForSequenceClassification
from deepoffense.classification.transformer_models.camembert_model import CamembertForSequenceClassification
from deepoffense.classification.transformer_models.distilbert_model import DistilBertForSequenceClassification
from deepoffense.classification.transformer_models.flaubert_model import FlaubertForSequenceClassification
from deepoffense.classification.transformer_models.roberta_model import RobertaForSequenceClassification
from deepoffense.classification.transformer_models.xlm_model import XLMForSequenceClassification
from deepoffense.classification.transformer_models.xlm_roberta_model import XLMRobertaForSequenceClassification
from deepoffense.classification.transformer_models.xlnet_model import XLNetForSequenceClassification
from deepoffense.custom_models.models import ElectraForSequenceClassification

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class ExplainableModel(ClassificationModel):
    def __init__(
            self,
            model_type,
            model_name,
            params,
            num_labels=None,
            weight=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            onnx_execution_provider=None,
            **kwargs,
    ):

        """
        Initializes a ExplainableModel model.
        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "auto": (AutoConfig, SC_weighted_BERT, AutoTokenizer),
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
            "bertweet": (
                RobertaConfig,
                RobertaForSequenceClassification,
                BertweetTokenizer,
            ),

            "camembert": (
                CamembertConfig,
                CamembertForSequenceClassification,
                CamembertTokenizerFast,
            ),
            "deberta": (
                DebertaConfig,
                DebertaForSequenceClassification,
                DebertaTokenizer,
            ),
            "distilbert": (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            ),
            "electra": (
                ElectraConfig,
                ElectraForSequenceClassification,
                ElectraTokenizerFast,
            ),
            "flaubert": (
                FlaubertConfig,
                FlaubertForSequenceClassification,
                FlaubertTokenizer,
            ),

            "longformer": (
                LongformerConfig,
                LongformerForSequenceClassification,
                LongformerTokenizerFast,
            ),

            "roberta": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
            ),

            "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForSequenceClassification,
                XLMRobertaTokenizerFast,
            ),
            "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)
        if self.args.labels_list:
            if num_labels:
                assert num_labels == len(self.args.labels_list)
            if self.args.labels_map:
                try:
                    assert list(self.args.labels_map.keys()) == self.args.labels_list
                except AssertionError:
                    assert [int(key) for key in list(self.args.labels_map.keys())] == self.args.labels_list
                    self.args.labels_map = {int(key): value for key, value in self.args.labels_map.items()}
            else:
                self.args.labels_map = {label: i for i, label in enumerate(self.args.labels_list)}
        else:
            len_labels_list = 2 if not num_labels else num_labels
            self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

            options = SessionOptions()
            options.intra_op_num_threads = 1

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(model_path.as_posix(), options, providers=[onnx_execution_provider])
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(model_path, options, providers=[onnx_execution_provider])
        else:
            if not self.args.quantized_model:
                if self.weight:
                    self.model = model_class.from_pretrained(
                        model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
                    )
                else:
                    print('load model')
                    self.model = model_class.from_pretrained(model_name, config=self.config, params=params, **kwargs)
            else:
                quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
                if self.weight:
                    self.model = model_class.from_pretrained(
                        None,
                        config=self.config,
                        state_dict=quantized_weights,
                        weight=torch.Tensor(self.weight).to(self.device),
                    )
                else:
                    self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        self.weight = None
        # TODO: compara with classification model

    def _load_model_args(self, input_dir):
        args = ClassificationArgs()  # TODO: pass explainability args as generic based on each model
        args.load(input_dir)
        return args

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "input_modal": batch[2],
            "attention_mask": batch[1],
            "modal_start_tokens": batch[3],
            "modal_end_tokens": batch[4],
        }

        return inputs

    def train_model(
            self,
            train_df,
            multi_label=False,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_df=None,
            verbose=True,
            **kwargs,
    ):
        return super().train_model(
            train_df,
            multi_label=multi_label,
            eval_df=eval_df,
            output_dir=output_dir,
            show_running_loss=show_running_loss,
            verbose=True,
            args=args,
            **kwargs,
        )

    def standaloneEval_with_rational(
            self,
            params,
            test_data=None,
            extra_data_path=None,
            topk=2,
            use_ext_df=False,
            multi_label=False,
            output_dir=None,
            verbose=True,
            silent=False,
            wandb_log=True,
            **kwargs
    ):

        """
          Evaluates the model on eval_df. Saves results to output_dir.
          Args:
              eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
              the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
              output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
              verbose: If verbose, results will be printed to the console on completion of evaluation.
              silent: If silent, tqdm progress bars will be hidden.
              wandb_log: If True, evaluation results will be logged to wandb.
              **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                          A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
          Returns:
              result: Dictionary containing evaluation results.
              model_outputs: List of model outputs for each row in eval_df
              wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
          """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        return self.evaluate(
            output_dir, params,
            test_data,
            extra_data_path,
            topk,
            use_ext_df,
            multi_label=multi_label, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
        )

    # TODO: rename
    def evaluate(
            self,
            output_dir,
            params,
            test_data=None,
            extra_data_path=None,
            topk=2,
            use_ext_df=False,
            multi_label=False,
            prefix="",
            verbose=True,
            silent=False,
            wandb_log=True,
            **kwargs
    ):

        """
        Evaluates the model on eval_df.
        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        print(args)
        eval_output_dir = output_dir

        test = createDatasetSplit(params)
        vocab_own = None

        results = {}

        if (extra_data_path != None):
            temp_read = pd.read_csv(params['data_file'], sep="\t")
            temp_read['raw_text'] = temp_read['text']
            temp_read.text = temp_read.tokens.str.split()
            temp_read.rationales = temp_read.rationales.apply(lambda x: [ast.literal_eval(x)])
            # print(temp_read.iloc[4]['rationales'])
            temp_read['final_label'] = temp_read['label']
            test_data = get_test_data(temp_read, params, self.tokenizer, message='text')
            test_extra = encodeData(test_data, vocab_own, params)
            eval_dataloader = combine_features(test_extra, params, is_train=False)
        elif (use_ext_df):
            test_extra = encodeData(test_data, vocab_own, params)
            eval_dataloader = combine_features(test_extra, params, is_train=False)
        else:
            eval_dataloader = combine_features(test, params, is_train=False)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(test), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(test), self.num_labels))
        else:
            out_label_ids = np.empty((len(test)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        if ((extra_data_path != None) or (use_ext_df == True)):
            post_id_all = list(test_data['Post_id'])
        else:
            post_id_all = list(test['Post_id'])

        true_labels = []
        pred_labels = []
        logits_all = []
        attention_all = []
        input_mask_all = []

        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation")):
            # batch = tuple(t.to(self.device) for t in batch)
            # print(batch)
            # b_input_ids, b_att_val, b_input_mask, b_labels
            with torch.no_grad():
                b_input_ids = batch[0].to(self.device)
                b_att_val = batch[1].to(self.device)
                b_input_mask = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)

                outputs = model(b_input_ids,
                                attention_vals=b_att_val,
                                attention_mask=b_input_mask,
                                labels=None, device=self.device)

                logits = outputs[0]  # logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy()  # out_label_ids

            attention_vectors = np.mean(outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1)

            pred_labels += list(np.argmax(logits, axis=1).flatten())
            true_labels += list(label_ids.flatten())
            logits_all += list(logits)
            attention_all += list(attention_vectors)
            input_mask_all += list(batch[2].detach().cpu().numpy())

        logits_all_final = []
        for logits in logits_all:
            logits_all_final.append(softmax(logits))

        if (use_ext_df == False):
            testf1 = f1_score(true_labels, pred_labels, average='macro')
            testacc = accuracy_score(true_labels, pred_labels)
            testprecision = precision_score(true_labels, pred_labels, average='macro')
            testrecall = recall_score(true_labels, pred_labels, average='macro')

            # Report the final accuracy for this validation run.
            print(" Accuracy: {0:.3f}".format(testacc))
            print(" Fscore: {0:.3f}".format(testf1))
            print(" Precision: {0:.3f}".format(testprecision))
            print(" Recall: {0:.3f}".format(testrecall))
            # print(" Test took: {:}".format(format_time(time.time() - t0)))

        attention_vector_final = []
        for x, y in zip(attention_all, input_mask_all):
            temp = []
            for x_ele, y_ele in zip(x, y):
                if (y_ele == 1):
                    temp.append(x_ele)
            attention_vector_final.append(temp)

        list_dict = []

        for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final,
                                                                  pred_labels, true_labels):
            temp = {}
            encoder = LabelEncoder()
            encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
            # print(encoder.classes_)
            pred_label = encoder.inverse_transform([pred])[0]

            temp["annotation_id"] = post_id
            temp["classification"] = pred_label
            temp["classification_scores"] = {"NOT": logits[0], "OFF": logits[1]}

            topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]

            temp_hard_rationales = []
            for ind in topk_indicies:
                temp_hard_rationales.append({'end_token': ind + 1, 'start_token': ind})

            temp["rationales"] = [{"docid": post_id,
                                   "hard_rationale_predictions": temp_hard_rationales,
                                   "soft_rationale_predictions": attention,
                                   "truth": ground_truth}]
            list_dict.append(temp)

        return list_dict, test_data

    def load_and_cache_examples(
            self,
            examples,
            evaluate=False,
            no_cache=False,
            multi_label=False,
            verbose=True,
            silent=False
    ):
        """
                Converts a list of example objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
                Utility function for train() and eval() methods. Not intended to be used directly.
                """
        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        cached_features_file = os.path.join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
            ),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                logger.info(" Converting to features started. Cache is not used.")
                if args.sliding_window:
                    logger.info(" Sliding window enabled")

            # If labels_map is defined, then labels need to be replaced with ints
            labels_map = {
                'NOT': 0,
                'OFF': 1
            }

            if not self.args.regression:
                for example in examples:
                    if multi_label:
                        example.label = [labels_map[label] for label in example.label]
                    else:
                        example.label = labels_map[example.label]

            features = convert_examples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args.silent or silent,
                use_multiprocessing=args.use_multiprocessing,
                sliding_window=args.sliding_window,
                flatten=not evaluate,
                stride=args.stride,
                add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # avoid padding in case of single example/online inferencing to decrease execution time
                pad_to_max_length=bool(len(examples) > 1),
                args=args,
            )
            if verbose and args.sliding_window:
                logger.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)

        if args.sliding_window and evaluate:
            features = [
                [feature_set] if not isinstance(feature_set, list) else feature_set for feature_set in features
            ]
            features = [feature for feature_set in features for feature in feature_set]

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_att_vals = torch.tensor([f.att_vals for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if self.args.model_type == "layoutlm":
            all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        if self.args.model_type == "layoutlm":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        return eval_dataloader

    def compute_metrics(
            self, preds, labels, eval_examples=None, multi_label=False, **kwargs
    ):
        return super().compute_metrics(
            preds,
            labels,
            eval_examples,
            multi_label=multi_label,
            **kwargs,
        )

    def predict(self, to_predict, multi_label=True):
        return super().predict(to_predict, multi_label=multi_label)

    def get_final_dict_with_rational(self, params, test_data=None, topk=5):
        list_dict_org, test_data = self.standaloneEval_with_rational(params, extra_data_path=test_data, topk=topk)
        test_data_with_rational = convert_data(test_data, params, list_dict_org, rational_present=True, topk=topk)
        list_dict_with_rational, _ = self.standaloneEval_with_rational(params, test_data=test_data_with_rational,
                                                                       topk=topk,
                                                                       use_ext_df=True)
        test_data_without_rational = convert_data(test_data, params, list_dict_org, rational_present=False, topk=topk)
        list_dict_without_rational, _ = self.standaloneEval_with_rational(params, test_data=test_data_without_rational,
                                                                          topk=topk, use_ext_df=True)
        final_list_dict = []
        for ele1, ele2, ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
            ele1['sufficiency_classification_scores'] = ele2['classification_scores']
            ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
            final_list_dict.append(ele1)
        return final_list_dict


