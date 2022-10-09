import logging
import os
import random
import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    BertweetTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
    LongformerConfig,
    LongformerTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
)


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from deepoffense.classification.classification_utils import LazyClassificationDataset, InputExample, \
    convert_examples_to_features, sweep_config_to_sweep_values

from deepoffense.classification import ClassificationModel
from deepoffense.classification.classification_utils import sweep_config_to_sweep_values
from deepoffense.classification.transformer_models.args.model_args import MultiLabelClassificationArgs, ClassificationArgs
from deepoffense.custom_models.models import AlbertForMultiLabelSequenceClassification, \
    BertweetForMultiLabelSequenceClassification, CamembertForMultiLabelSequenceClassification, \
    DistilBertForMultiLabelSequenceClassification, ElectraForMultiLabelSequenceClassification, \
    FlaubertForMultiLabelSequenceClassification, LongformerForMultiLabelSequenceClassification, \
    RobertaForMultiLabelSequenceClassification, XLMForMultiLabelSequenceClassification, \
    XLMRobertaForMultiLabelSequenceClassification, XLNetForMultiLabelSequenceClassification, \
    BertForMultiLabelSequenceClassification

from tqdm.auto import tqdm, trange
from dataclasses import asdict
from scipy.stats import mode

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
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from deepoffense.classification.classification_utils import LazyClassificationDataset, InputExample, \
    convert_examples_to_features, sweep_config_to_sweep_values
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
            "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
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
                    self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
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
        #TODO: compara with classification model

    def _load_model_args(self, input_dir):
        args = ClassificationArgs() #TODO: pass explainability args as generic based on each model
        args.load(input_dir)
        return args

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

    def eval_model(
        self, 
        eval_df, 
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

      result, model_outputs, wrong_preds = self.evaluate(
          eval_df, output_dir, multi_label=multi_label, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
      )
      self.results.update(result)

      if verbose:
          logger.info(self.results)

      return result, model_outputs, wrong_preds

    def evaluate(
        self, 
        eval_df, 
        output_dir, 
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
        eval_output_dir = output_dir

        results = {}
        if isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError("Lazy loading is not implemented for LayoutLM models")
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError("Input must be given as a path to a file when using lazy loading")

            if "text" in eval_df.columns and "labels" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    eval_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                eval_df["text"].astype(str),
                                eval_df["labels"],
                                eval_df["x0"],
                                eval_df["y0"],
                                eval_df["x1"],
                                eval_df["y1"],
                            )
                        )
                    ]
                else:
                    eval_examples = [
                        InputExample(i, text, None, label)
                        for i, (text, label) in enumerate(zip(eval_df["text"].astype(str), eval_df["labels"]))
                    ]
            elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    eval_examples = [
                        InputExample(i, text_a, text_b, label)
                        for i, (text_a, text_b, label) in enumerate(
                            zip(eval_df["text_a"].astype(str), eval_df["text_b"].astype(str), eval_df["labels"])
                        )
                    ]
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                eval_examples = [
                    InputExample(i, text, None, label)
                    for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
                ]

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
        # os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), self.num_labels))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation")):
            # batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = inputs["labels"].detach().cpu().numpy()

            # if preds is None:
            #     preds = logits.detach().cpu().numpy()
            #     out_label_ids = inputs["labels"].detach().cpu().numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.sliding_window:
            count = 0
            window_ranges = []
            for n_windows in window_counts:
                window_ranges.append([count, count + n_windows])
                count += n_windows

            preds = [preds[window_range[0]: window_range[1]] for window_range in window_ranges]
            out_label_ids = [
                out_label_ids[i] for i in range(len(out_label_ids)) if i in [window[0] for window in window_ranges]
            ]

            model_outputs = preds

            preds = [np.argmax(pred, axis=1) for pred in preds]
            final_preds = []
            for pred_row in preds:
                mode_pred, counts = mode(pred_row)
                if len(counts) > 1 and counts[0] == counts[1]:
                    final_preds.append(args.tie_value)
                else:
                    final_preds.append(mode_pred[0])
            preds = np.array(final_preds)
        elif not multi_label and args.regression is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds

            if not multi_label:
                preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if self.args.wandb_project and wandb_log and not multi_label and not self.args.regression:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            if not args.labels_map:
                self.args.labels_map = {i: i for i in range(self.num_labels)}

            labels_list = sorted(list(self.args.labels_map.keys()))
            inverse_labels_map = {value: key for key, value in self.args.labels_map.items()}

            truth = [inverse_labels_map[out] for out in out_label_ids]

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, [inverse_labels_map[pred] for pred in preds], labels=labels_list,
            )

            if not self.args.sliding_window:
                # ROC`
                wandb.log({"roc": wandb.plots.ROC(truth, model_outputs, labels_list)})

                # Precision Recall
                wandb.log({"pr": wandb.plots.precision_recall(truth, model_outputs, labels_list)})

        return results, model_outputs, wrong

    def load_and_cache_examples(
        self, 
        examples, 
        evaluate=False, 
        no_cache=False, 
        multi_label=False, 
        verbose=True, 
        silent=False
    ):
        
        return super().load_and_cache_examples(
            examples,
            evaluate=evaluate,
            no_cache=no_cache,
            multi_label=multi_label,
            verbose=verbose,
            silent=silent,
        )

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