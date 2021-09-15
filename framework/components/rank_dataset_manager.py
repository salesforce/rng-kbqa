"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


# Generation dataset manager
# Managing generation dataset load/cache and prediction
import torch
import os

from os.path import join
from components.dataset_utils import ListDataset
from components.rank_dataset import (
    grail_read_examples_from_jsonline_file,
    webqsp_read_examples_from_jsonline_file,
    extract_features_from_examples,
)

def grail_load_and_cache_rank_examples(args, tokenizer, evaluate=False):
    logger = args.logger
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]
    condition_id = os.path.splitext(os.path.basename(split_file))[0].split('-')[-1]
    cached_features_file = os.path.join('feature_cache',"{}_{}_{}_{}_{}_{}".format(dataset_id,
        split_id, condition_id, args.model_type, args.linear_method.split('_')[0], args.max_seq_length))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file
        orig_split = split_id
        dataset_file = join('outputs', f'grailqa_v1.0_{orig_split}.json')
        examples = grail_read_examples_from_jsonline_file(dataset_file, candidate_file, args.linear_method, is_eval=evaluate)
        features = extract_features_from_examples(args, tokenizer, examples)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    return ListDataset(features)

def webqsp_load_and_cache_rank_examples(args, tokenizer, evaluate=False):
    logger = args.logger
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]
    condition_id = os.path.splitext(os.path.basename(split_file))[0].split('-')[-1]
    cached_features_file = os.path.join('feature_cache',"{}_{}_{}_{}_{}_{}".format(dataset_id,
        split_id, condition_id, args.model_type, args.linear_method.split('_')[0], args.max_seq_length))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file
        # TODO: hard coded for now
        orig_split = split_id
        dataset_file = join('outputs', f'WebQSP.{orig_split}.expr.json')
        examples = webqsp_read_examples_from_jsonline_file(dataset_file, candidate_file, args.linear_method, is_eval=evaluate)
        features = extract_features_from_examples(args, tokenizer, examples)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    return ListDataset(features)


def load_and_cache_rank_examples(args, tokenizer, evaluate=False):
    if args.dataset == 'grail':
        return grail_load_and_cache_rank_examples(args, tokenizer, evaluate=evaluate)
    elif args.dataset == 'webqsp':
        return webqsp_load_and_cache_rank_examples(args, tokenizer, evaluate=evaluate)
    else:
        raise RuntimeError('Unsupported Ranking Dataset')
