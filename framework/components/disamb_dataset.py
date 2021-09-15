"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import json
from re import A
import types
import torch
import numpy as np

from torch.utils.data import Dataset
from types import SimpleNamespace
from tqdm import tqdm
from collections import Counter

from components.utils import *
from components.grail_utils import extract_mentioned_entities
from executor.sparql_executor import get_label, execute_query, get_in_relations, get_out_relations
from executor.logic_form_util import lisp_to_sparql
from nltk.tokenize import word_tokenize


class _MODULE_DEFAULT:
    IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld']
    RELATION_FREQ_FILE = 'misc/relation_freq.json'
    RELATION_FREQ = None

class GrailEntityCandidate:
    def __init__(self, id, label, facc_label, surface_score, pop_score, relations):
        self.id = id
        self.label = label
        self.facc_label = facc_label
        self.surface_score = surface_score
        self.pop_score = pop_score
        self.relations = relations

    def __str__(self):
        # return self.id + ':' + self.label
        return '{}:{}:{:.2f}'.format(self.id, self.label, self.surface_score)

    def __repr__(self):
        return self.__str__()

# For on single mention
# We link to only one entity for each mention
class GrailEntityDisambProblem:
    def __init__(self, pid, query, mention, target_id, candidates):
        self.pid = pid # problem id
        self.qid = pid.split('-')[0]
        self.query = query
        self.mention = mention
        self.target_id = target_id
        self.candidates = candidates

# a instance containing multiple problems to be solved
class GrailEntityDisambInstance:
    def __init__(self, qid, query, s_expr, target_entities, disamb_problems):
        self.qid = qid
        self.query = query
        self.s_expr = s_expr
        self.target_entities = target_entities
        self.target_labels = [get_label(x) for x in target_entities]
        self.disamb_problems = disamb_problems

# for single problem
class GrailEntityDisambFeature:
    def __init__(self, pid, input_ids, token_type_ids, target_idx):
        self.pid = pid
        self.candidate_input_ids = input_ids
        self.candidate_token_type_ids = token_type_ids
        self.target_idx = target_idx

def proc_instance(ex, linking_results, cutoff = 10):
    qid = str(ex['qid'])
    query = ex['question']
    if 's_expression' in ex:
        s_expr = ex['s_expression']
        entities_in_gt = set(extract_mentioned_entities(s_expr))
    else:
        s_expr = 'null'
        entities_in_gt = set()
    
    ranking_problems = []
    # topk_set = set( chain(*[[x['id'] for x in entities_per_mention[:cutoff]] for entities_per_mention in linking_results]) )
    # if entities_in_gt.issubset(topk_set):
    for idx, entities_per_metion in enumerate(linking_results):
        entities_per_metion = entities_per_metion[:cutoff]

        # no linked entty in this mention
        if not entities_per_metion:
            continue

        entities_included = set([e['id'] for e in entities_per_metion])
        candidates = []
        for entity in entities_per_metion:
            eid = entity['id']
            fb_label = get_label(eid)
            in_relations = get_in_relations(eid)
            out_relations = get_out_relations(eid)
            # print(fb_label, in_relations, out_relations)
            candidates.append(GrailEntityCandidate(
                entity['id'], fb_label, entity['label'],
                entity['surface_score'], entity['pop_score'], 
                in_relations | out_relations))

        target = next((x for x in entities_included if x in entities_in_gt), None)
        problem_id = f'{qid}-{idx}'
        single_problem = GrailEntityDisambProblem(problem_id, query, entities_per_metion[0]['mention'], target, candidates)
        ranking_problems.append(single_problem)
    entity_ex = GrailEntityDisambInstance(qid, query, s_expr, entities_in_gt, ranking_problems)
    return entity_ex

def _tokenize_relation(r):
    return r.replace('.', ' ').replace('_', ' ').split()

def _normalize_relation(r):
    r = r.replace('_', ' ')
    r = r.replace('.', ' , ')
    return r

def _construct_disamb_context(args, tokenizer, candidate, proc_query_tokens):
    if _MODULE_DEFAULT.RELATION_FREQ is None:
        _MODULE_DEFAULT.RELATION_FREQ = load_json(_MODULE_DEFAULT.RELATION_FREQ_FILE)
    relations = [x for x in candidate.relations if x.split('.')[0] not in _MODULE_DEFAULT.IGONORED_DOMAIN_LIST]
    # print(relations)
    # return relations
    def key_func(r):
        r_tokens = _tokenize_relation(r)
        overlapping_val = len(set(proc_query_tokens) & set(r_tokens))
        return(
            _MODULE_DEFAULT.RELATION_FREQ.get(r, 1),
            -overlapping_val
            )
    relations = sorted(relations, key=lambda x: key_func(x))

    relations_str = ' ; '.join(map(_normalize_relation, relations))
    return relations_str

def _extract_disamb_feature_from_problem(args, tokenizer, problem):
    pid = problem.pid
    query = problem.query
    query_tokens = word_tokenize(query.lower())

    candidate_input_ids = []
    candidate_token_type_ids = []
    if args.do_lower_case:
        query = query.lower()
    for c in problem.candidates:
        relation_info = _construct_disamb_context(args, tokenizer, c, query_tokens)
        label_info = c.label
        if label_info is None:
            label_info = ''
            print('WANING INVALID LABEL', c.id, '|', c.label, '|', c.facc_label)
        if args.do_lower_case:
            relation_info = relation_info.lower()
            label_info = label_info.lower()
        context_info = '{} {} {}'.format(label_info, tokenizer.sep_token, relation_info)
        
        c_encoded = tokenizer(query, context_info, truncation=True, max_length=args.max_seq_length, return_token_type_ids=True)
        candidate_input_ids.append(c_encoded['input_ids'])
        candidate_token_type_ids.append(c_encoded['token_type_ids'])

        # print(tokenizer.convert_ids_to_tokens(candidate_input_ids[-1]))
        # print(context_info)
    target_idx = next((i for (i,x) in enumerate(problem.candidates) if x.id == problem.target_id), 0)
    return GrailEntityDisambFeature(pid, candidate_input_ids, candidate_token_type_ids, target_idx)


def _collect_contrastive_inputs(feat, num_sample, dummy_inputs):
    input_ids = []
    token_type_ids = []
    sample_mask = []
    
    input_ids.extend(feat.candidate_input_ids)
    token_type_ids.extend(feat.candidate_token_type_ids)
    filled_num = len(input_ids)
    # force padding
    for _ in range(filled_num, num_sample):
        input_ids.append(dummy_inputs['input_ids'])
        token_type_ids.append(dummy_inputs['token_type_ids'])
    sample_mask = [1] * filled_num + [0] * (num_sample - filled_num)
    return input_ids, token_type_ids, sample_mask

def disamb_collate_fn(data, tokenizer):
    dummy_inputs = tokenizer('','', return_token_type_ids=True)
    # batch size
    # input_id: B * N_Sample * L
    # token_type: B * N_Sample * L
    # attention_mask: B * N_Sample * N
    # sample_mask: B * N_Sample
    # labels: B, all zero
    batch_size = len(data)
    num_sample = max([len(x.candidate_input_ids) for x in data])

    all_input_ids = []
    all_token_type_ids = []
    all_sample_masks = []
    for feat in data:
        input_ids, token_type_ids, sample_mask = _collect_contrastive_inputs(feat, num_sample, dummy_inputs)
        all_input_ids.extend(input_ids)
        all_token_type_ids.extend(token_type_ids)
        all_sample_masks.append(sample_mask)
    
    encoded = tokenizer.pad({'input_ids': all_input_ids, 'token_type_ids': all_token_type_ids},return_tensors='pt')
    all_sample_masks = torch.BoolTensor(all_sample_masks)
    labels = torch.LongTensor([x.target_idx for x in data])
    
    all_input_ids = encoded['input_ids'].view((batch_size, num_sample, -1))
    all_token_type_ids = encoded['token_type_ids'].view((batch_size, num_sample, -1))
    all_attention_masks = encoded['attention_mask'].view((batch_size, num_sample, -1))
    return all_input_ids, all_token_type_ids, all_attention_masks, all_sample_masks, labels

def read_disamb_instances_from_entity_candidates(dataset_file, candidate_file):
    dataset = load_json(dataset_file)
    entity_linking_results = load_json(candidate_file)

    instances = []
    dataset = dataset
    for data in tqdm(dataset, total=len(dataset), desc='Read Exapmles'):
        qid = str(data['qid'])
        res = entity_linking_results[qid]
        instances.append(proc_instance(data, res))

    return instances

# it only returns **VALID FEATURES**
def extract_disamb_features_from_examples(args, tokenizer, instances, do_predict=False):
    valid_disamb_problems = []
    baseline_acc = 0
    for inst in instances:
        for p in inst.disamb_problems:
            # print(len(p.ca))
            if not do_predict:
                if (len(p.candidates) > 1) and p.target_id is not None:
                # if (len(p.candidates) > 0) and p.target_id is not None:
                    valid_disamb_problems.append(p)
                    if p.candidates[0].id == p.target_id:
                        baseline_acc += 1
            else:
                if (len(p.candidates) > 1):
                    valid_disamb_problems.append(p)

    hints = 'VALID: {}, ACC: {:.1f}'.format(len(valid_disamb_problems), baseline_acc / len(valid_disamb_problems))
    features = []
    for p in tqdm(valid_disamb_problems, total=len(valid_disamb_problems), desc=hints):
        feat = _extract_disamb_feature_from_problem(args, tokenizer, p)
        features.append(feat)
    return features

def coverage_evaluation(instances, valid_features, predicted_indexes):
    # build result index
    indexed_pred = dict([(feat.pid, pred) for feat, pred in zip(valid_features, predicted_indexes)])
    # for (feat, pred) in zip(valid_features, predicted_indexes):

    covered = 0
    for inst in instances:
        gt_entities = inst.target_entities
        pred_entities = []
        for problem in inst.disamb_problems:
            if len(problem.candidates) == 0:
                continue
            if len(problem.candidates) == 1 or problem.target_id is None:
                pred_entities.append(problem.candidates[0].id) 
                continue

            pred_idx = indexed_pred[problem.pid]
            pred_entities.append(problem.candidates[pred_idx].id)
        # print(gt_entities, pred_entities)
        if set(gt_entities).issubset(set(pred_entities)):
            covered += 1
    coverage = covered / len(instances)
    return coverage


def precompute_relation_frequency():
    # instances = read_disamb_instances('train')
    instances = load_bin('train_entities.bin')
    all_relations = []
    for ex in instances:
        for problem in ex.disamb_problems:
            for candidate in problem.candidates:
                all_relations.extend(candidate.relations)
    relation_counter = Counter(all_relations)
    print(len(relation_counter))
    dump_json(relation_counter, _MODULE_DEFAULT.RELATION_FREQ_FILE)
    for k, v in relation_counter.most_common(100):
        print(k, v)
    # return relation_counter

# unit test
def test_coverage_evaluation():
    data = torch.load('feature_cache/disamb_grail_dev_bert_96')
    valid_features = data['features']
    examples = data['examples']
    print(sum([x.target_idx == 0 for x in valid_features]) / len(valid_features))
    print(coverage_evaluation(examples, valid_features, [0] * len(valid_features)))

def read_disamb_instances(split):
    dataset = load_json(f'outputs/grailqa_v1.0_{split}.json')
    entity_linking_results = load_json(f'outputs/grail_{split}-entities.json')

    instances = []
    dataset = dataset
    print(len(dataset), len(entity_linking_results))
    for data in tqdm(dataset, total=len(dataset)):
        qid = str(data['qid'])
        res = entity_linking_results[qid]
        instances.append(proc_instance(data, res))

    return instances

def test_read_instances():
    instances = read_disamb_instances('dev')
    dump_to_bin(instances, 'dev_entities.bin')
    # for instance in instances:
    #     print(instance.qid)
    #     print(instance.query)
    #     print(instance.s_expr)
    #     print('Target', list(zip(instance.target_entities, instance.target_labels)))
    #     print('num mention', len(instance.disamb_problems))
    #     for prob in instance.disamb_problems:
    #         print('\t', prob.pid, prob.mention,  prob.candidates)
    #         if prob.candidates:
    #             print(prob.candidates[0].relations)

    instances = load_bin('mini_entities.bin')
    valid_disamb_problems = []
    for inst in instances:
        for p in inst.disamb_problems:
            if (len(p.candidates) > 1) and p.target_id is not None:
            # if (len(p.candidates) > 0) and p.target_id is not None:
                valid_disamb_problems.append(p)
    baseline_acc = 0
    for p in valid_disamb_problems:
        if p.candidates[0].id == p.target_id:
            baseline_acc += 1
    print('BASELLINE ACC', baseline_acc / len(valid_disamb_problems), len(valid_disamb_problems))

    # encode the features
    
    features = []
    from types import SimpleNamespace
    args = SimpleNamespace()
    # args.max_seq_length = 96
    # from transformers.tokenization_auto import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # args.do_lower_case = True
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    args.do_lower_case = False
    for p in valid_disamb_problems[:1]:
        feat = _extract_disamb_feature_from_problem(args, tokenizer, p)
        features.append(feat)