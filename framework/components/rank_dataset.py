"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import json
import types
import torch
import numpy as np

from torch.utils.data import Dataset
from types import SimpleNamespace
from tqdm import tqdm

from components.utils import *
from components.expr_parser import textualize_s_expr
from components.dataset_utils import LFCandidate

from components.legacy_value_extractor import GrailQA_Value_Extractor
from executor.sparql_executor import get_label, execute_query
from executor.logic_form_util import lisp_to_sparql

class RankingExample:
    def __init__(self, qid, query, gt, candidates, entity_label_map, friendly_name_map={}, level='null', answers=[]):
        self.qid = qid
        self.query = query
        self.gt = gt
        self.candidates = candidates
        self.friendly_name_map = friendly_name_map
        self.entity_label_map = entity_label_map
        self.level = level
        self.answers = answers

    def __str__(self):
        return '{}\n\t->{}\n'.format(self.query, self.gt.normed_expr)

    def __repr__(self):
        return self.__str__()

class RankingFeature:
    def __init__(self, qid, ex, gt_input_ids, gt_token_typ_ids, candidate_input_ids, candidate_token_type_ids):
        self.qid = qid
        self.ex = ex
        self.gt_input_ids = gt_input_ids
        self.gt_token_type_ids = gt_token_typ_ids
        self.candidate_input_ids = candidate_input_ids
        self.candidate_token_type_ids = candidate_token_type_ids

def resolve_entity_map(qid, query, el_results, el_extractor):
    _delimiter=";"
    # el results currently only contain dev and test results, skipped for now
    # TODO: entity_map is not used for now and skipped for not 
    if not qid in el_results:
        return {}
    entity_map = el_results[qid]['entities']
    entities = set(entity_map.keys())
    for k in entity_map:
        v = entity_map[k]['friendly_name']
        entity_map[k] = ' '.join(v.replace(_delimiter, ' ').split()[:5])
    literals = set()
    mentions = el_extractor.detect_mentions(query)
    for m in mentions:
        literals.add(el_extractor.process_literal(m))
    return entity_map

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

def _naive_textualization_method(expr, entity_label_map):
    textual_form = textualize_s_expr(expr)
    toks = textual_form.split(' ')
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
        else:
            if '_' in t:
                t = t.replace('_', ' ')
            if '.' in t:
                t = t.replace('.', ' , ')

        # normalize type
        norm_toks.append(t)
    return ' '.join(norm_toks)

def _reduced_texutalization_method(expr, entity_label_map):
    textual_form = textualize_s_expr(expr)
    toks = textual_form.split(' ')
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
        elif '.' in t:
            meta_relations = t = t.split('.')
            t = meta_relations[-1]
            if '_' in t:
                t = t.replace('_', ' ')
        # normalize type
        norm_toks.append(t)
    return ' '.join(norm_toks)

def _normalize_s_expr(expr, entity_label_map, linear_method):
    if linear_method == 'vanilla':
        return _vanilla_linearization_method(expr, entity_label_map)
    elif linear_method == 'naive_text':
        return _naive_textualization_method(expr, entity_label_map)
    elif linear_method == 'reduct_text':
        return _reduced_texutalization_method(expr, entity_label_map)
    else:
        raise RuntimeWarning('Unsupported linearization method')

def grail_proc_ranking_exs(line, data_bank, linear_method, is_eval=False):
    candidates_info = json.loads(line)
    
    qid = candidates_info['qid']
    raw_data = data_bank[qid]
    # fields
    # ['qid', 's_expression', 'candidates']
    # ['qid', 'question', 'answer', 'domains', 'level', 's_expression']

    query = raw_data['question']
    # gt_expr = raw_data['s_expression']
    gt_expr = candidates_info['target_expr']
    raw_candidates = candidates_info['candidates']
    answer = []

    entity_label_map = {} # resolve_entity_label(qid, gt, candidates)
    norm_gt = _normalize_s_expr(gt_expr, entity_label_map, linear_method)
    # print('normed gt', norm_gt)
    gt = LFCandidate(gt_expr, norm_gt, True, 1.0, 0.0)
    candidates = []
    for c in raw_candidates:
        c_expr = c['logical_form']
        normed_c_expr = _normalize_s_expr(c_expr, entity_label_map, linear_method)
        # print('normed c_expr', normed_c_expr)
        c_ex = c['ex']
        lf_candidate = LFCandidate(c_expr, normed_c_expr, c_ex)
        candidates.append(lf_candidate)

    return RankingExample(qid, query, gt, candidates, entity_label_map)

def grail_read_examples_from_jsonline_file(dataset_file, candidate_file, linear_method='vanilla', is_eval=False):
    data_bank = load_json(dataset_file)

    data_bank = dict([(str(x['qid']), x) for x in data_bank])
    with open(candidate_file) as f:
        lines = f.readlines()
    examples = []
    for l in tqdm(lines, desc='Reading', total=len(lines)):
        ex = grail_proc_ranking_exs(l, data_bank, linear_method=linear_method, is_eval=False)
        examples.append(ex)
    return examples

def webqsp_proc_ranking_exs(candidates_info, data_bank, linear_method, is_eval=False):
    qid = candidates_info['qid']
    raw_data = data_bank[qid]
    # fields
    # ['qid', 's_expression', 'candidates']
    # ['qid', 'question', 'answer', 'domains', 'level', 's_expression']

    query = raw_data['RawQuestion']
    gt_expr = candidates_info['target_s_expr']
    raw_candidates = candidates_info['candidates']
    # control, do not train on invalid data
    if is_eval:
        if len(raw_candidates) == 0:
            return None
    else:
        if gt_expr == 'null' or len(raw_candidates) == 0:
            return None
    answer = []

    entity_label_map = {} # resolve_entity_label(qid, gt, candidates)
    norm_gt = _normalize_s_expr(gt_expr, entity_label_map, linear_method)
    # print('normed gt', norm_gt)
    gt = LFCandidate(gt_expr, norm_gt, True, 1.0, 0.0)
    candidates = []
    for c in raw_candidates:
        c_expr = c['logical_form']
        normed_c_expr = _normalize_s_expr(c_expr, entity_label_map, linear_method)
        # print('normed c_expr', normed_c_expr)
        c_ex = c['ex']
        lf_candidate = LFCandidate(c_expr, normed_c_expr, c_ex)
        candidates.append(lf_candidate)
    return RankingExample(qid, query, gt, candidates, entity_label_map)


def webqsp_read_examples_from_jsonline_file(dataset_file, candidate_file, linear_method='vanilla', is_eval=False):
    data_bank = load_json(dataset_file)
    data_bank = dict([(str(x['QuestionId']), x) for x in data_bank])
    lines = load_json(candidate_file)
    examples = []
    for l in tqdm(lines, desc='Reading', total=len(lines)):
        ex = webqsp_proc_ranking_exs(l, data_bank, linear_method=linear_method, is_eval=is_eval)
        if ex is None:
            continue
        examples.append(ex)
    return examples

def _extract_feature_from_example(args, tokenizer, ex):
    # gt_input_ids, gt_token_type_ids, candidates_input_ids, candidates_token_type_ids
    qid = ex.qid
    q = ex.query
    gt_lf = ex.gt.normed_expr

    if args.do_lower_case:
        q = q.lower()
        gt_lf = gt_lf.lower()
    gt_encoded = tokenizer(q, gt_lf, truncation=True, max_length=args.max_seq_length, return_token_type_ids=True)
    gt_input_ids, gt_token_type_ids = gt_encoded['input_ids'], gt_encoded['token_type_ids']
    # print(gt_lf)
    # print(tokenizer.convert_ids_to_tokens(gt_input_ids))
    # exit()
    candidate_input_ids, candidate_token_type_ids = [], []
    for c in ex.candidates:
        c_lf = c.normed_expr
        if args.do_lower_case:
            c_lf = c_lf.lower()
        c_encoded = tokenizer(q, c_lf, truncation=True, max_length=args.max_seq_length, return_token_type_ids=True)
        candidate_input_ids.append(c_encoded['input_ids'])
        candidate_token_type_ids.append(c_encoded['token_type_ids'])
    
    return RankingFeature(qid, ex, gt_input_ids, gt_token_type_ids, candidate_input_ids, candidate_token_type_ids)
    
def extract_features_from_examples(args, tokenizer, examples):
    features = []
    for ex in tqdm(examples, desc='Indexing', total=len(examples)):
        feat = _extract_feature_from_example(args, tokenizer, ex)
        features.append(feat)
    return features

def _collect_contrastive_inputs(feat, num_sample, dummy_inputs, selected_negative):
    input_ids = []
    token_type_ids = []
    sample_mask = []
    input_ids.append(feat.gt_input_ids)
    token_type_ids.append(feat.gt_token_type_ids)
    
    for idx in selected_negative:
        input_ids.append(feat.candidate_input_ids[idx])
        token_type_ids.append(feat.candidate_token_type_ids[idx])
    
    filled_num = len(input_ids)
    # force padding
    for _ in range(filled_num, num_sample):
        input_ids.append(dummy_inputs['input_ids'])
        token_type_ids.append(dummy_inputs['token_type_ids'])
    sample_mask = [1] * filled_num + [0] * (num_sample - filled_num)
    return input_ids, token_type_ids, sample_mask

def collect_random_contrastiveness(feat, num_sample, dummy_inputs):

    num_negative = num_sample - 1 
    negative_ids = [i for (i,x) in enumerate(feat.ex.candidates) if not x.ex]
    if len(negative_ids) > num_negative:
        selected_negative = np.random.choice(negative_ids, num_negative, replace=False)
    else:
        selected_negative = negative_ids

    return _collect_contrastive_inputs(feat, num_sample, dummy_inputs, selected_negative)

def collect_boostrap_contrastiveness(feat, num_sample, dummy_inputs):   
    num_negative = num_sample - 1 
    negative_ids = [i for (i,x) in enumerate(feat.ex.candidates) if not x.ex]
    if len(negative_ids) > num_negative:
        negative_ids.sort(key=lambda x: feat.ex.candidates[x].score, reverse=True)
        selected_negative = negative_ids[:num_negative]
    else:
        selected_negative = negative_ids
    # for i in selected_negative:
    #     print(feat.ex.candidates[i].score, feat.ex.candidates[i].ex)
    #     print(feat.ex.candidates[i].normed_expr)
    #     print(feat.ex.gt.normed_expr)
    return _collect_contrastive_inputs(feat, num_sample, dummy_inputs, selected_negative)

def collect_mixbootstrap_contrastiveness(feat, num_sample, dummy_inputs, bs_ratio=0.5):
    num_negative = num_sample - 1 
    negative_ids = [i for (i,x) in enumerate(feat.ex.candidates) if not x.ex]
    if len(negative_ids) > num_negative:
        num_bs = int(num_negative * bs_ratio)
        negative_ids.sort(key=lambda x: feat.ex.candidates[x].score, reverse=True)
        selected_bs_negative = negative_ids[:num_bs]
        num_rand = num_negative - num_bs
        selected_rand_negative = np.random.choice(negative_ids[num_bs:], num_rand, replace=False)
        selected_negative = selected_bs_negative + selected_rand_negative.tolist()
    else:
        selected_negative = negative_ids
    # print(len(selected_negative), 'TOTAL', len(negative_ids))
    # for i in selected_negative:
    #     print(i, feat.ex.candidates[i].score, feat.ex.candidates[i].ex)
    #     # print(feat.ex.candidates[i].normed_expr)
    #     # print(feat.ex.gt.normed_expr)

    return _collect_contrastive_inputs(feat, num_sample, dummy_inputs, selected_negative)

def contrastive_collate_fn(data, tokenizer, num_sample, strategy='random'):
    dummy_inputs = tokenizer('','', return_token_type_ids=True)
    # batch size
    # input_id: B * N_Sample * L
    # token_type: B * N_Sample * L
    # attention_mask: B * N_Sample * N
    # sample_mask: B * N_Sample
    # labels: B, all zero
    batch_size = len(data)

    all_input_ids = []
    all_token_type_ids = []
    all_sample_masks = []
    for feat in data:
        if strategy == 'random':
            input_ids, token_type_ids, sample_mask = collect_random_contrastiveness(feat, num_sample, dummy_inputs)
        elif strategy == 'boostrap':
            input_ids, token_type_ids, sample_mask = collect_boostrap_contrastiveness(feat, num_sample, dummy_inputs)
        elif strategy == 'mixboostrap':
            input_ids, token_type_ids, sample_mask = collect_mixbootstrap_contrastiveness(feat, num_sample, dummy_inputs)
        else:
            raise RuntimeError('Unspported negative sampling')
        
        all_input_ids.extend(input_ids)
        all_token_type_ids.extend(token_type_ids)
        all_sample_masks.append(sample_mask)
    
    encoded = tokenizer.pad({'input_ids': all_input_ids, 'token_type_ids': all_token_type_ids},return_tensors='pt')
    all_sample_masks = torch.BoolTensor(all_sample_masks)
    labels = torch.LongTensor([0] * batch_size)
    
    all_input_ids = encoded['input_ids'].view((batch_size, num_sample, -1))
    all_token_type_ids = encoded['token_type_ids'].view((batch_size, num_sample, -1))
    all_attention_masks = encoded['attention_mask'].view((batch_size, num_sample, -1))
    return all_input_ids, all_token_type_ids, all_attention_masks, all_sample_masks, labels

def eval_iteretor_for_single_instance(feat, tokenizer):
    encoded = tokenizer.pad({'input_ids': feat.candidate_input_ids, 'token_type_ids': feat.candidate_token_type_ids},return_tensors='pt')
    isomorphic_labels = torch.LongTensor([x.ex for x in feat.ex.candidates])

    return encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask'], isomorphic_labels

def write_prediction_file(features, predicted_indexes, filename):
    # jsonlines
    # qid, logical_form, answer
    lines = []
    for f, idx in zip(features, predicted_indexes):
        qid = f.qid
        lf = f.ex.candidates[idx].s_expr
        # answer = 
        try:
            sparql_query = lisp_to_sparql(lf)
            denotation = execute_query(sparql_query)
        except:
            denotation = []
        lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer': denotation}))
    with open(filename, 'w') as f:
        f.writelines([x+'\n' for x in lines])

# TOOD: Different strategy of collate fn
# TODO: multi process acceleration
if __name__ == '__main__':
    pass