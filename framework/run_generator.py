"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import json
import logging
import os
import sys
import timeit
from os.path import join
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Iterable
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler


from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartTokenizer,
    HfArgumentParser,
    T5Tokenizer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    AutoModelForSeq2SeqLM
)

from components.utils import dump_json
from components.gen_dataset_manager import load_and_cache_examples, ListDataset
from components.gen_dataset import generation_collate_fn
from components.generation_trainer import GenerationTrainer
logger = logging.getLogger(__name__)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    warmup_ratio: Optional[float] = field(default=0.0, metadata={"help": "The warmup ratio"})
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(
        metadata={"help": "type of the model, t5 or bart"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset: str = field(default=None,metadata={"help": "dataset id"})
    train_file: str = field(default=None,metadata={"help": "path to training file"})
    predict_file: str = field(default=None,metadata={"help": "path to predict file"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='hfcache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    # freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    # freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    max_source_length: Optional[int] = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    top_k_candidates: Optional[int] = field(default=5, metadata={"help": "# top k candidates used for generation."})
    do_lower_case: bool = field(default=False)
    overwrite_cache: bool = field(default=False)

    # local_rank: Optional[int] = field(default=-1,metadata={"help": "local_rank for distributed training on gpus."})

def _pad_tensors_to_max_len(tensor, max_length, pad_token_id):
    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor

def run_prediction(args, dataset, model, tokenizer, output_prediction=False):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # For evaluating, we ensure batch size is only one
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size, collate_fn=partial(generation_collate_fn, tokenizer=tokenizer))

    max_length = (
        model.config.max_generate_length
        if hasattr(model.config, "max_generate_length")
        else model.config.max_position_embeddings
    )
    num_beams = model.config.num_beams
    pad_token_id = model.config.pad_token_id
    # multi-gpu evaluate
    # only allow using one gpu here
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.per_device_eval_batch_size)

    start_time = timeit.default_timer()
    model = model.to(args.device)

    all_predictions = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = dict([(k, v.to(args.device)) for k, v in batch.items()])

        labels = batch.pop('labels')
        [all_labels.append(l.cpu().numpy()) for l in labels]
        with torch.no_grad():
            generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    max_length=max_length,
                )
            generated_tokens = torch.reshape(generated_tokens, (labels.size(0), num_beams, -1))
            [all_predictions.append(p.cpu().numpy()) for p in generated_tokens]
            # in case the batch is shorter than max length, the output should be padded
        # for seq in generated_tokens:
        #     print(seq)
    assert len(all_predictions) == len(all_labels)

    ex_cnt = 0
    pred_outputs = OrderedDict()
    for feat, pred in tqdm(zip(dataset, all_predictions), total=len(all_predictions), desc='Decoding'):
        ex = feat.ex
        decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        ex_cnt += (decoded_pred[0] == ex.gt.normed_expr.replace(' ,', ',').lower())
        # print(decoded_pred[0])
        # print(ex.gt.normed_expr)
        pred_outputs[ex.qid] = decoded_pred

    if output_prediction:
        dump_json(pred_outputs, join(args.output_dir, 'top_k_predictions.json'))
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    return  {'total': len(all_predictions), 'ex': ex_cnt / len(all_predictions)}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    model_args.local_rank = training_args.local_rank
    # model_args.output_dir = training_args.output_dir
    # model_args.n_gpu = training_args.n_gpu
    # model_args.eval_batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # use task specific params
    # use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if model_args.eval_beams is not None:
        model.config.num_beams = model_args.eval_beams
    assert model.config.num_beams >= 1, f"got eval_beams={model.config.num_beams}. Need an integer >= 1"
    model_args.logger = logger

    # set max length for generation
    model.config.max_generate_length = model_args.max_target_length

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        # with decoding
        def _exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            pred_str, label_str = decode_pred(pred)
            ex = sum([a == b for (a,b) in zip(pred_str, label_str)])/len(pred_str)
            result = {'ex': ex}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result
        
        # without decoding
        def exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            # pred_str, label_str = decode_pred(pred)
            ex = np.sum(np.all(pred.label_ids == pred.predictions, axis=1)) / pred.label_ids.shape[0]
            # for a, b in zip(pred.label_ids, pred.predictions):
            #     print(a)
            #     print(b)
            # exit()
            result = {'ex': ex, 'num_total': pred.label_ids.shape[0]}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
        compute_metrics_fn = exact_match_metrics
        return compute_metrics_fn


    # Get datasets
    if training_args.do_train:
        train_dataset = ListDataset(load_and_cache_examples(model_args, tokenizer, evaluate=False))
    else:
        train_dataset = ListDataset([])
    if training_args.do_eval:
        eval_dataset = ListDataset(load_and_cache_examples(model_args, tokenizer, evaluate=True))
    else:
        eval_dataset = ListDataset([])

    # Training
    if training_args.do_train:
        # Initialize our Trainer
        trainer = GenerationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=partial(generation_collate_fn, tokenizer=tokenizer),
            # prediction_loss_only=True
            compute_metrics=build_compute_metrics_fn(),
        )

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # prediction
    eval_results = {}
    if training_args.do_eval:
        logging.info("*** Test ***")

        result = run_prediction(training_args, eval_dataset, model, tokenizer, output_prediction=True)
        # if trainer.is_world_process_zero():
        logger.info("***** Test results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)

        eval_results.update(result)
    return eval_results


if __name__ == "__main__":
    main()