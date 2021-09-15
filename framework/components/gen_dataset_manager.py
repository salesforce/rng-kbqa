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

from torch.utils.data import Dataset
from components.gen_dataset import (
    webqsp_read_gen_examples_from_json,
    grail_read_gen_examples_from_json,
    extract_gen_features_from_examples,
)

class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)


def webqsp_load_and_cache_gen_examples(args, tokenizer, evaluate=False):
    logger = args.logger
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]
    # split_file = '_'.(join(os.path.basename(split_file).split('_')[:2])
    cached_features_file = os.path.join('feature_cache',"gen_{}_{}_{}_{}".format(dataset_id,
        split_id, args.model_type, args.top_k_candidates))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file
        dataset_file = join('outputs', f'WebQSP.{split_id}.expr.json')
        examples = webqsp_read_gen_examples_from_json(dataset_file, candidate_file, is_eval=evaluate)
        features = extract_gen_features_from_examples(args, tokenizer, examples)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    return ListDataset(features)

def grail_load_and_cache_gen_examples(args, tokenizer, evaluate=False):
    logger = args.logger
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]
    # split_file = '_'.(join(os.path.basename(split_file).split('_')[:2])
    cached_features_file = os.path.join('feature_cache',"gen_{}_{}_{}_{}".format(dataset_id,
        split_id, args.model_type, args.top_k_candidates))
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file
        dataset_file = join('outputs', f'grailqa_v1.0_{split_id}.json')
        examples = grail_read_gen_examples_from_json(dataset_file, candidate_file, is_eval=evaluate)
        features = extract_gen_features_from_examples(args, tokenizer, examples)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    return ListDataset(features)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.dataset == 'grail':
        return grail_load_and_cache_gen_examples(args, tokenizer, evaluate=evaluate)
    elif args.dataset == 'webqsp':
        return webqsp_load_and_cache_gen_examples(args, tokenizer, evaluate=evaluate)
    else:
        raise RuntimeError('Unsupported Generation Dataset')
