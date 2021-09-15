"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from collections import OrderedDict
import torch
from components.utils import *
from tqdm import tqdm
import re
from executor.sparql_executor import get_label, execute_query
from executor.logic_form_util import lisp_to_sparql, same_logical_form
import json
from components.expr_parser import extract_entities, tokenize_s_expr
import argparse
import spacy

# copied from grail value extractor
def process_literal(value: str):  # process datetime mention; append data type
    pattern_date = r"(?:(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
    pattern_datetime = r"\d{4}-\d{2}-\d{2}t[\d:z-]+"
    pattern_float = r"(?:[-]*\d+[.]*\d*e[+-]\d+|(?<= )[-]*\d+[.]\d*|^[-]*\d+[.]\d*)"
    pattern_yearmonth = r"\d{4}-\d{2}"
    pattern_year = r"(?:(?<= )\d{4}|^\d{4})"
    pattern_int = r"(?:(?<= )[-]*\d+|^[-]*\d+)"
    if len(re.findall(pattern_datetime, value)) == 1:
        value = value.replace('t', "T").replace('z', 'Z')
        return f'{value}^^http://www.w3.org/2001/XMLSchema#dateTime'
    elif len(re.findall(pattern_date, value)) == 1:
        if value.__contains__('-'):
            return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
        elif value.__contains__('/'):
            fields = value.split('/')
            value = f"{fields[2]}-{fields[0]}-{fields[1]}"
            return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
    elif len(re.findall(pattern_yearmonth, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#gYearMonth'
    elif len(re.findall(pattern_float, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#float'
    elif len(re.findall(pattern_year, value)) == 1 and int(value) <= 2015:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#gYear'
    elif len(re.findall(pattern_int, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#integer'
    else:
        return 'null'

def is_value_tok(t):
    if t[0].isalpha():
        return False
    return (process_literal(t) != 'null')

def denomarlize_s_expr(normed_expr, entity_label_map):
    expr = normed_expr
    for e, v in entity_label_map.items():
        expr = expr.replace(v + ' )', e + ')')
        expr = expr.replace(v.lower() + ' )', e + ')')
    p0 = expr
    expr = expr.replace('( greater equal', '( ge')
    expr = expr.replace('( greater than', '( gt')
    expr = expr.replace('( less equal', '( le')
    expr = expr.replace('( less than', '( lt')

    expr = expr.replace(', ', ' , ')
    toks = expr.split(' ')
    
    # adhoc replace 'g,'
    # new_toks = []
    # for i, t in enumerate(toks[:-2]):
    #     if t == 'g,' and toks[i + 2] == ')':
    #         new_toks.append('g#proc')
    #     else:
    #         new_toks.append(t)
    # expr = ' '.join(new_toks)
    # expr = expr.replace('g#proc ', 'g.')
    # toks = expr.split(' ')

    # print(torch.exp)
    # syntext_checker = ''
    # print(toks)
    prev_left_par = False
    segments = []
    cur_seg = ''
    for t in toks:
        if t == '(':
            prev_left_par = True
            if cur_seg:
                segments.append(cur_seg)
            segments.append(t)
            cur_seg = ''
            continue
        # if prev_left_par
        if prev_left_par:
            if t in ['ge', 'gt', 'le', 'lt']:
                segments.append(t)
            else:                
                segments.append(t.upper())
            prev_left_par = False
            cur_seg = ''
            continue
        prev_left_par = False
        if t == ')':
            if cur_seg:
                segments.append(cur_seg)
            segments.append(t)
            cur_seg = ''
            continue
        elif t.startswith('m.') or t.startswith('g.'):
            if cur_seg:
                segments.append(cur_seg)
            segments.append(t)
            cur_seg = ''
            continue
        elif t == ',':
            cur_seg += '.'
            continue
        elif is_value_tok(t):
            if cur_seg:
                segments.append(cur_seg)
            proc_t = process_literal(t)
            segments.append(proc_t)
            cur_seg = ''
            continue
        else:
            if cur_seg and cur_seg[-1] != '.':
                cur_seg += f'_{t}'
            else:
                cur_seg += t
            continue
    expr = ' '.join(segments)
    return expr

def execute_normed_s_expr(normed_expr, entity_label_map):
    try:
        denormed_expr = denomarlize_s_expr(normed_expr, entity_label_map)
    except:
        return 'null', []
    query_expr = denormed_expr

    query_expr = query_expr.replace('( ', '(').replace(' )', ')')
    if query_expr != 'null':
        try:
            # print('parse', query_expr)
            sparql_query = lisp_to_sparql(query_expr)
            # print('sparql', sparql_query)
            denotation = execute_query(sparql_query)
        except:
            denotation = []
    # print(query_expr, denotation)
    return query_expr, denotation

def execute_vanilla_s_expr(s_expr):
    try:
        # print('parse', query_expr)
        sparql_query = lisp_to_sparql(s_expr)
        # print('sparql', sparql_query)
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    return denotation

def get_entity_mapping_from_top_candidates(feat):
    logical_forms = feat['top_candidates']
    logical_forms = [x['logical_form'] for x in logical_forms]

    entity_label_map = OrderedDict()
    for lf in logical_forms:
        entities = extract_entities(lf)
        for e in entities:
            if e not in entity_label_map:
                entity_label_map[e] = get_label(e)
    return entity_label_map

def aggressive_top_k_eval(split, predict_file):
    predictions = load_json(predict_file)
    generation_dataset = load_json(f'outputs/grail_{split}_gen.json')

    ex_cnt = 0
    top_hit = 0
    lines = []
    
    gen_exectuable_cnt = 0
    final_exectuable_cnt = 0
    for feat in tqdm(generation_dataset, total=len(generation_dataset)):
        qid = feat['qid']
        pred = predictions[qid]

        entity_label_map = get_entity_mapping_from_top_candidates(feat)
        found_exectutable = False

        for rank, p in enumerate(pred):
            lf, answers = execute_normed_s_expr(p, entity_label_map)

            if rank == 0 and lf == feat['genation_target']:
                ex_cnt += 1
            # print(answers)
            if answers:
                if len(answers) == 1 and answers[0] == '0':
                    continue
                lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer': answers,}))
                found_exectutable = True
                if rank == 0:
                    top_hit += 1
                break

        if found_exectutable:
            gen_exectuable_cnt += 1


        if not found_exectutable and len(feat['top_candidates']):
            for can in feat['top_candidates']:
                query_expr = can['logical_form']
                try:
                    sparql_query = lisp_to_sparql(query_expr)
                    denotation = execute_query(sparql_query)
                except:
                    denotation = []
                if denotation:
                    if len(denotation) == 1 and denotation[0] == '0':
                        continue
                    lines.append(json.dumps({'qid': qid, 'logical_form': query_expr, 'answer': denotation}))
                    found_exectutable = True
                    break
        if found_exectutable:
            final_exectuable_cnt += 1

    print('STR Match', ex_cnt / len(generation_dataset))
    print('TOP 1 Executable', top_hit / len(generation_dataset))
    print('Gen Executable', gen_exectuable_cnt / len(generation_dataset))
    print('Final Executable', final_exectuable_cnt / len(generation_dataset))

    with open(f'misc/grail_{split}_gen_results.txt', 'w') as f:
        f.writelines([x+'\n' for x in lines])

def top_k_upperbound(split, predict_file):
    predictions = load_json(predict_file)
    generation_dataset = load_json(f'outputs/grail_{split}_gen.json')
    dataset = load_json('outputs/grailqa_v1.0_dev.json')
    dataset = dict([(str(x['qid']), x) for x in dataset])

    ex_cnt = 0
    top_hit = 0
    lines = []
    
    first_ex_results = []
    best_ex_results = []
    for feat in tqdm(generation_dataset, total=len(generation_dataset)):
        qid = feat['qid']
        pred = predictions[qid]

        entity_label_map = get_entity_mapping_from_top_candidates(feat)
        gt_lf = dataset[qid]['s_expression']
        gt_answers = set()
        if dataset[qid]['answer'] != 'null':
            for a in dataset[qid]['answer']:
                gt_answers.add(a['answer_argument'])

        found_exectutable = False
        best_res = (0,0)
        for rank, p in enumerate(pred):
            lf, answers = execute_normed_s_expr(p, entity_label_map)

            if rank == 0 and lf == feat['genation_target']:
                ex_cnt += 1
            # print(answers)
            if answers:
                lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer': answers,}))
                predict_answer = set(answers)
                em = same_logical_form(lf, gt_lf)    
                if em:
                    f1 = 1.0
                else:
                    if len(predict_answer.intersection(gt_answers)) != 0:
                        precision = len(predict_answer.intersection(gt_answers)) / len(predict_answer)
                        recall = len(predict_answer.intersection(gt_answers)) / len(gt_answers)
                        f1 = (2 * recall * precision / (recall + precision))
                    else:
                        f1 = 0
                if f1 > best_res[1]:
                    best_res = (em, f1)

                if not found_exectutable:
                    found_exectutable = True
                    first_ex_results.append((em, f1))
                if best_res[0] == 1.0 or best_res[1] == 1.0:
                    break
        best_ex_results.append(best_res)
        if not found_exectutable:
            first_ex_results.append(best_res)

            
    print('First')
    print(sum([x[0] for x in first_ex_results]) / len(first_ex_results), sum([x[1] for x in first_ex_results]) / len(first_ex_results), len(first_ex_results))
    print('Best')
    print(sum([x[0] for x in best_ex_results]) / len(best_ex_results), sum([x[1] for x in best_ex_results]) / len(best_ex_results), len(best_ex_results))

def force_top_1_eval(split, predict_file):
    predictions = load_json(predict_file)
    generation_dataset = load_json(f'outputs/grail_{split}_gen.json')

    lines = []

    for feat in tqdm(generation_dataset, total=len(generation_dataset)):
        qid = feat['qid']
        pred = predictions[qid]

        entity_label_map = get_entity_mapping_from_top_candidates(feat)

        top_pred = pred[0]
        lf, answers = execute_normed_s_expr(top_pred, entity_label_map)

        lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer': answers,}))


    with open(f'misc/grail_{split}_gen_top_results.txt', 'w') as f:
        f.writelines([x+'\n' for x in lines])


def read_grail_prediction(filename):
    with open(filename) as f:
        lines = f.readlines()
    predictions = [json.loads(x) for x in lines]
    return predictions



class NNEmptyEntityPredictor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.ref = self.construct_reference()
        self.ref =  [x + (self.nlp(x[0]),) for x in self.ref]

    def construct_reference(self):
        dataset = load_json('outputs/grailqa_v1.0_train.json')
        empty_pairs = []
        for i, d in enumerate(dataset):
            s_expr = d['s_expression']
            toks = tokenize_s_expr(s_expr)
            entities = [x for x in toks if x.startswith('m.') or x.startswith('g.')]
            if '^^http://' in s_expr:
                continue
            if d['answer'] == 'null':
                continue
            if len(entities) == 0:
                answer_set = [a['answer_argument'] for a in d['answer']]
                empty_pairs.append((d['question'], d['s_expression'], answer_set))
        return empty_pairs

    def predict(self, qid, query):
        scores = []
        x = self.nlp(query)
        for i, y in enumerate(self.ref):
            s = x.similarity(y[-1])
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        predicted_idx = scores[0][0]

        pred = self.ref[predicted_idx]
        return {'qid': qid, 'logical_form': pred[1], 'answer': pred[2]}
        
def remedy_unincluded_by_nn(split):
    dataset = load_json(f'outputs/grailqa_v1.0_{split}.json') 
    current_predictions = read_grail_prediction(f'misc/grail_{split}_gen_results.txt')
    covered_ids = [x['qid'] for x in current_predictions]

    nn_revision = NNEmptyEntityPredictor()
    revise_num = 0
    for d in tqdm(dataset, total=len(dataset)):
        qid = str(d['qid'])
        if qid in covered_ids:
            continue
        
        revise_num += 1
        pred = nn_revision.predict(qid, d['question'])
        # print(d['question'])
        # print(pred)
        current_predictions.append(pred)
    print(revise_num)

    lines = [json.dumps(x) for x in current_predictions]
    with open(f'misc/grail_{split}_final_results.txt', 'w') as f:
        f.writelines([x+'\n' for x in lines])

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--pred_file', default=None, help='topk prediction file')
    parser.add_argument('--revise_only', action='store_true', dest='revise_only', default=False, help='only do revising')
    args = parser.parse_args()
    if args.pred_file is None:
        args.pred_file = f'misc/grail_{args.split}_topk_generations.json'

    print('split', args.split, 'topk_file', args.pred_file)
    return args

if __name__=='__main__':
    args = _parse_args()
    # top_k_eval(args.split, args.pred_file)
    # force_top_1_eval(args.split, args.pred_file)
    # top_k_upperbound(args.split, args.pred_file)
    if not args.revise_only:
        aggressive_top_k_eval(args.split, args.pred_file)
        remedy_unincluded_by_nn(args.split)
    else:
        remedy_unincluded_by_nn(args.split)