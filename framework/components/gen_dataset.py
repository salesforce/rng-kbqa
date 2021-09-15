"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from components.utils import *
import os
from components.dataset_utils import LFCandidate
from executor.sparql_executor import get_label
from tqdm import tqdm

from transformers import BartTokenizer

class GenerationExample:
    def __init__(self, qid, query, gt, candidates, entity_label_map, answers=[]):
        self.qid = qid
        self.query = query
        self.gt = gt
        self.candidates = candidates
        self.entity_label_map = entity_label_map
        self.answers = answers

    def __str__(self):
        return '{}\n\t->{}\n'.format(self.query, self.gt.normed_expr)

    def __repr__(self):
        return self.__str__()

class GenerationFeature:
    def __init__(self, ex, src_input_ids, tgt_input_ids):
        self.ex = ex
        self.src_input_ids = src_input_ids
        self.tgt_input_ids = tgt_input_ids

def _vanilla_linearization_method(expr, entity_label_map):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]

    norm_toks = []
    for t in toks:
        # normalize entity
        if t.startswith('m.'):
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                name = get_label(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
        elif 'XMLSchema' in t:
            format_pos = t.find('^^')
            t = t[:format_pos]
        elif t == 'ge':
            t = 'GREATER EQUAL'
        elif t == 'gt':
            t = 'GREATER THAN'
        elif t == 'le':
            t = 'LESS EQUAL'
        elif t == 'lt':
            t = 'LESS THAN'
        else:
            if '_' in t:
                t = t.replace('_', ' ')
            if '.' in t:
                t = t.replace('.', ' , ')

        # normalize type
        norm_toks.append(t)
    return ' '.join(norm_toks)


def proc_webqsp_gen_exs(candidates_info, data_bank):
    qid = candidates_info['qid']
    raw_data = data_bank[qid]

    query = raw_data['RawQuestion']
    gt_expr = candidates_info['genation_target']
    entity_label_map = {} # resolve_entity_label(qid, gt, candidates)
    norm_gt = _vanilla_linearization_method(gt_expr, entity_label_map)
    # print('normed gt', norm_gt)
    gt = LFCandidate(gt_expr, norm_gt, True, 1.0, 0.0)
    top_candidates = candidates_info['top_candidates']
    candidates = []
    for c in top_candidates:
        c_expr = c['logical_form']
        normed_c_expr = _vanilla_linearization_method(c_expr, entity_label_map)
        # print('normed c_expr', normed_c_expr)
        c_ex = c['ex']
        lf_candidate = LFCandidate(c_expr, normed_c_expr, c_ex)
        candidates.append(lf_candidate)

    return GenerationExample(qid, query, gt, candidates, entity_label_map, answers=[])


def webqsp_read_gen_examples_from_json(dataset_file, candidate_file, is_eval=False):
    data_bank = load_json(dataset_file)
    data_bank = dict([(str(x['QuestionId']), x) for x in data_bank])
    lines = load_json(candidate_file)
    examples = []
    for l in tqdm(lines, desc='Reading', total=len(lines)):
        ex = proc_webqsp_gen_exs(l, data_bank)
        if ex is None:
            continue
        examples.append(ex)
    return examples


def proc_grail_gen_exs(candidates_info, data_bank):
    qid = candidates_info['qid']
    raw_data = data_bank[qid]

    query = raw_data['question']
    gt_expr = candidates_info['genation_target']
    entity_label_map = {} # resolve_entity_label(qid, gt, candidates)
    norm_gt = _vanilla_linearization_method(gt_expr, entity_label_map)
    # print('normed gt', norm_gt)
    gt = LFCandidate(gt_expr, norm_gt, True, 1.0, 0.0)
    top_candidates = candidates_info['top_candidates']
    candidates = []
    for c in top_candidates:
        c_expr = c['logical_form']
        normed_c_expr = _vanilla_linearization_method(c_expr, entity_label_map)
        # print('normed c_expr', normed_c_expr)
        c_ex = c['ex']
        lf_candidate = LFCandidate(c_expr, normed_c_expr, c_ex)
        candidates.append(lf_candidate)

    return GenerationExample(qid, query, gt, candidates, entity_label_map, answers=[])


def grail_read_gen_examples_from_json(dataset_file, candidate_file, is_eval=False):
    data_bank = load_json(dataset_file)
    data_bank = dict([(str(x['qid']), x) for x in data_bank])

    lines = load_json(candidate_file)
    examples = []
    for l in tqdm(lines, desc='Reading', total=len(lines)):
        ex = proc_grail_gen_exs(l, data_bank)
        if ex is None:
            continue
        examples.append(ex)
    return examples

def _extract_gen_feature_from_example(args, tokenizer, ex, add_prefix_space=False):
    # gt_input_ids, gt_token_type_ids, candidates_input_ids, candidates_token_type_ids
    qid = ex.qid
    q = ex.query
    gt_lf = ex.gt.normed_expr

    if args.do_lower_case:
        q = q.lower()
        gt_lf = gt_lf.lower()

    candidate_lfs = []
    for c in ex.candidates[:args.top_k_candidates]:
        c_lf = c.normed_expr
        if args.do_lower_case:
            c_lf = c_lf.lower()
        candidate_lfs.append(c_lf)

    src_text = ' ; '.join([q] + candidate_lfs)
    dst_text = gt_lf
    if add_prefix_space:
        batch_encoding = tokenizer.prepare_seq2seq_batch(
                [src_text],
                [dst_text],
                max_length=args.max_source_length,
                max_target_length=args.max_target_length,
                return_tensors="pt",
                add_prefix_space=add_prefix_space,
            ).data
    else:
        batch_encoding = tokenizer.prepare_seq2seq_batch(
                [src_text],
                [dst_text],
                max_length=args.max_source_length,
                max_target_length=args.max_target_length,
                return_tensors="pt",
            ).data
        # batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        # return batch_encoding    # return GrailRankingFeature(qid, ex, gt_input_ids, gt_token_type_ids, candidate_input_ids, candidate_token_type_ids)
    input_ids, labels = batch_encoding['input_ids'][0], batch_encoding['labels'][0]
    # encoded = tokenizer.pad({'input_ids': [input_ids, input_ids[:20]]},return_tensors='pt')
    # encoded = tokenizer.pad({'input_ids': [labels, labels[:5]]},return_tensors='pt')
    return GenerationFeature(ex, input_ids, labels)

def generation_collate_fn(data, tokenizer):
    all_input_ids = []
    all_labels = []
    for feat in data:
        all_input_ids.append(feat.src_input_ids)
        all_labels.append(feat.tgt_input_ids)

    src_encoded = tokenizer.pad({'input_ids': all_input_ids},return_tensors='pt')
    tgt_encoded = tokenizer.pad({'input_ids': all_labels},return_tensors='pt')
    return {
        'input_ids': src_encoded['input_ids'],
        'attention_mask': src_encoded['attention_mask'],
        'labels': tgt_encoded['input_ids']
    }

def extract_gen_features_from_examples(args, tokenizer, examples):
    features = []
    add_prefix_space = isinstance(tokenizer, BartTokenizer)
    for ex in tqdm(examples, desc='Indexing', total=len(examples)):
        feat = _extract_gen_feature_from_example(args, tokenizer, ex, add_prefix_space=add_prefix_space)
        features.append(feat)
    return features
