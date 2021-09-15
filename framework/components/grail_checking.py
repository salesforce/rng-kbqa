"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


# for sanity check and determing some hyper-parameters
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from types import SimpleNamespace
from tqdm import tqdm

from components.utils import *
from entity_linking.value_extractor import GrailQA_Value_Extractor
from executor.sparql_executor import get_label
from components.grail_dataset import read_examples_from_jsonline_file, extract_features_from_examples, contrastive_collate_fn

# sanity check features
def sanity_check_feature(tokenizer, features):
    # checklist: max len
    # unk?
    max_len = 0
    unk_count = 0
    encoded_lens = []
    for f in tqdm(features, desc='Checking', total=len(features)):
        max_len = max(max_len, len(f.gt_input_ids))
        encoded_lens.append(len(f.gt_input_ids))
        for c_input_ids in f.candidate_input_ids:
            max_len = max(max_len, len(c_input_ids))
        if tokenizer.unk_token_id:
            flag = False
            if tokenizer.unk_token_id in f.gt_input_ids:
                flag = True
            for c_input_ids in f.candidate_input_ids:
                if tokenizer.unk_token_id in c_input_ids:
                    flag = True
            if flag:
                unk_count += 1
                print(f.ex.query, f.ex.gt.normed_expr)
                print(f.qid, tokenizer.convert_ids_to_tokens(f.gt_input_ids))

    print(max_len)
    print(unk_count)
    return encoded_lens

def sanity_check_encoding():
    examples = read_examples_from_jsonline_file('outputs/grailqa_v1.0_dev.json', 'outputs/grail_mini_candidates-edist.jsonline')
    # examples = read_examples_from_jsonline_file('outputs/grailqa_v1.0_dev.json', 'outputs/grail_dev_candidates-edist.jsonline')
    dump_to_bin(examples, 'tmpexs.bin')
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    examples = load_bin('tmpexs.bin')
    args = SimpleNamespace()
    args.lowercase = True
    features = extract_features_from_examples(args, tokenizer, examples)
    torch.save(features, 'tmpfeat.bin')

    features = torch.load('tmpfeat.bin')
    sanity_check_feature(tokenizer, features)

def sanity_check_random_collate():
    features = torch.load('tmpfeat.bin')
    # sanity_check_feature(tokenizer, features)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    args = SimpleNamespace()
    args.lowercase = True
    
    sample_slice = features[:2]
    for x in sample_slice:
        x.ex.candidates = x.ex.candidates[:2]
        print(len(x.candidate_input_ids))
        print([c.ex for c in x.ex.candidates])
        print(x.ex.gt.normed_expr)
        for c in x.ex.candidates:
            print(c.normed_expr)
    all_input_ids, all_token_type_ids, all_attention_masks, all_sample_masks, labels = contrastive_collate_fn(sample_slice, tokenizer, 4)
    print(all_input_ids.size())
    print(all_token_type_ids.size())
    print(all_attention_masks.size())
    print(all_sample_masks.size())
    print(labels.size())

    print(all_input_ids)
    print(all_token_type_ids)
    print(all_attention_masks)
    print(all_sample_masks)
    print(labels)

def checking_feature_entry():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # features = torch.load('feature_cache/grail_dev_bert_256') 
    # features = torch.load('feature_cache/grail_mini_bert_256') 
    features = torch.load('feature_cache/grail_mini_bert_32')
    encoded_lens = sanity_check_feature(tokenizer, features)

    frac = lambda t:  sum([x <= t for x in encoded_lens]) / len(encoded_lens)
    
    for l in [16, 32, 64,96,128,160,192,256]:
        print(l, frac(l))
    
    # selected = torch.threshold

def make_stress_dataset():
    # features = torch.load('feature_cache/grail_dev_bert_256') 
    features = torch.load('feature_cache/grail_dev_bert_96') 
    encoded_lens = [len(x.gt_input_ids) for x in features]

    features.sort(key=lambda x: len(x.gt_input_ids), reverse=True)
    features = features[:16]
    print([len(x.gt_input_ids) for x in features])
    torch.save(features, 'feature_cache/grail_stress_bert_96')

def checking_candidates_coverage():
    # examples = read_examples_from_jsonline_file('outputs/grailqa_v1.0_dev.json', 'outputs/grail_mini_candidates-edist.jsonline')
    # examples = read_examples_from_jsonline_file('outputs/grailqa_v1.0_dev.json', 'outputs/grail_dev_candidates-edist.jsonline')
    examples = read_examples_from_jsonline_file('outputs/grailqa_v1.0_dev.json', 'outputs/grail_dev_candidates-edist-perfectel.jsonline')


    example_dict = dict([(x.qid, x) for x in examples])
    from collections import Counter
    raw_examples = load_json('outputs/grailqa_v1.0_dev.json')
    hit_cnt = Counter()
    num_cnt = Counter()
    for raw_ex in raw_examples:
        level = raw_ex['level']
        num_cnt.update([level])
        num_cnt
        qid = str(raw_ex['qid'])
        if qid not in example_dict:
            continue
        ex = example_dict[qid]
        if any([x.ex for x in ex.candidates]):
            hit_cnt.update([level])
    
    print('agg', sum(hit_cnt.values())/sum(num_cnt.values()))
    for k in hit_cnt:
        print(k, hit_cnt[k] / num_cnt[k])
