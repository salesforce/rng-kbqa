"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import torch
from components.utils import *
from enumerate_candidates import get_approx_s_expr
from tqdm import tqdm
import re
from executor.sparql_executor import get_label, execute_query
from executor.logic_form_util import lisp_to_sparql
from collections import OrderedDict
import json
import argparse
from components.expr_parser import extract_entities, parse_s_expr, tokenize_s_expr

YEAR_DIGIT_RE = re.compile('\d{4}')


# MACRO mined from the training split, using extract_macro_template('train') in parse_sparql.py
MACRO_TEMPLATES = [('american_football.football_historical_coach_position', 'from', 'to'), ('architecture.ownership', 'start_date', 'end_date'),
 ('award.award_honor', 'year', 'year'), ('business.employment_tenure', 'from', 'to'), ('business.sponsorship', 'from', 'to'),
 ('celebrities.romantic_relationship', 'start_date', 'end_date'), ('chemistry.chemical_element', 'discovery_date', 'discovery_date'),
 ('film.film', 'initial_release_date', 'initial_release_date'),('government.government_position_held', 'from', 'to'),
 ('law.invention', 'date_of_invention', 'date_of_invention'), ('law.judicial_tenure', 'from_date', 'to_date'),
 ('organization.organization_relationship', 'to', 'from'), ('people.marriage', 'from', 'to'),
 ('people.place_lived', 'end_date', 'start_date'), ('sports.sports_team_coach_tenure', 'from', 'to'),
 ('sports.sports_team_roster', 'from', 'to'), ('sports.team_venue_relationship', 'from', 'to'),
 ('time.event', 'start_date', 'end_date'), ('tv.regular_tv_appearance', 'from', 'to'), ('tv.tv_network_duration', 'from', 'to')]
MACRO_TEMPLATES = dict([(x[0], (x[1], x[2])) for x in MACRO_TEMPLATES])

LEVEL_MACRO_CLAUSES = '''
FILTER(NOT EXISTS {{{var} ns:{relation}.{start_suffix} ?sk6}} || 
EXISTS {{{var} ns:{relation}.{start_suffix} ?sk7 . 
FILTER(xsd:datetime(?sk7) <= "{year}-12-31"^^xsd:dateTime) }})
FILTER(NOT EXISTS {{{var} ns:{relation}.{end_suffix} ?sk8}} || 
EXISTS {{{var} ns:{relation}.{end_suffix} ?sk9 . 
FILTER(xsd:datetime(?sk9) >= "{year}-01-01"^^xsd:dateTime) }})
'''

# type constraints mined from the training split, using mine_common_type_constraint('train') in parse_sparql.py
MINED_TYPE_CONSTRATES = [('m.0kpys4', 'US State'), ('m.044801x', 'Professional Sports Team'), ('m.01xljyt', 'American Football team'),
('m.01m9', 'City/Town/Village'), ('m.01xpjyz', 'Airport'), ('m.025dnr9', 'American Football Conference'), ('m.01xs05k', 'River'),
('m.01xryvm', 'Book'), ('m.01mh', 'Continent'), ('m.01y2hnl', 'College/University'),('m.01xljv1', 'Super bowl'), ('m.01xxv5b', 'Island Group'),
('m.02_3pws', 'Mexican state'), ('m.025dnqw', 'American Football Division'), ('m.01y2hn6', 'School'), ('m.01n7', 'Location'),
('m.03jz7ls', 'Written Work'), ('m.08scbsj', 'Subatomic particle'), ('m.03w5clp', 'Production company'), ('m.0kpym_', 'US County'),
('m.01xljtp', 'Hospital'), ('m.04fnrhx', 'Monarch'), ('m.01xs039', 'Mountain range'), ('m.01mp', 'Country'), ('m.02knxyp', 'Religious Text'),
('m.0256985', 'Baseball Team'), ('m.05czz29', 'Brand'), ('m.01nt', 'Region'), ('m.02ht342', 'Automobile Make'), ('m.02_3phk', 'Dutch province')]
MINED_TYPE_CONSTRATES = dict([(a, b.lower()) for (a, b) in MINED_TYPE_CONSTRATES])


def denomarlize_s_expr(normed_expr, entity_label_map):
    expr = normed_expr
    for e, v in entity_label_map.items():
        if v is not None:
            expr = expr.replace(v + ' )', e + ')')
            expr = expr.replace(v.lower() + ' )', e + ')')
    # apply mined type constraint
    # example
    # 'join common, topic, notable types school'
    # 'join common, topic, notable types m.01y2hn6'
    for e, v in MINED_TYPE_CONSTRATES.items():
        expr = expr.replace('join common, topic, notable types ' + v, 'join common, topic, notable types ' + e)

    expr = expr.replace(', ', ' , ')
    toks = expr.split(' ')
    
    # syntext_checker = ''
    # print(toks)
    prev_left_par = False
    segments = []
    cur_seg = ''
    for t in toks:
        if t == '(':
            prev_left_par = True
            segments.append(t)
            cur_seg = ''
            continue
        # if prev_left_par
        if prev_left_par:
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
        elif t.startswith('m.'):
            if cur_seg:
                segments.append(cur_seg)
            segments.append(t)
            cur_seg = ''
            continue
        elif YEAR_DIGIT_RE.match(t):
            if cur_seg:
                segments.append(cur_seg)
            segments.append(t + '^^http://www.w3.org/2001/XMLSchema#date')
            cur_seg = ''
            continue
        elif t == ',':
            cur_seg += '.'
            continue
        else:
            if cur_seg and cur_seg[-1] != '.':
                cur_seg += f'_{t}'
            else:
                cur_seg += t
            continue
    expr = ' '.join(segments)
    return expr



def _get_time_macro_clause(node):
    # print('NODE', node.construction, node.logical_form())
    if (node.construction == 'AND' and
        node.fields[0].construction == 'JOIN' and
        node.fields[0].fields[0].construction == 'SCHEMA' and 
        'time_macro' in node.fields[0].fields[0].val):
        return node.fields[0]
    else:
        for field in node.fields:
            ret_val = _get_time_macro_clause(field)
            if ret_val is not None:
                return ret_val
        return None

def get_time_macro_clause(x):
    assert '.time_macro' in x

    ast = parse_s_expr(x)
    macro_node = _get_time_macro_clause(ast)
    assert macro_node is not None
    
    relation = macro_node.fields[0].val
    year =  macro_node.fields[1].val[:4]
    assert relation.endswith('.time_macro')
    relation = '.'.join(relation.split('.')[:2])
    start_end_suffix = MACRO_TEMPLATES.get(relation, ('from', 'to'))
    # if macro_node
    if macro_node.level == 1:
        var = '?x'
    else:
        var = '?x0'

    additional_clause = LEVEL_MACRO_CLAUSES.format(var=var, relation=relation, start_suffix=start_end_suffix[0], end_suffix=start_end_suffix[1], year=year)
    return additional_clause

def execute_vanilla_s_expr(s_expr):
    try:
        # print('parse', query_expr)
        sparql_query = lisp_to_sparql(s_expr)
        # print('sparql', sparql_query)
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    return denotation

def execute_normed_s_expr(normed_expr, entity_label_map):
    denormed_expr = denomarlize_s_expr(normed_expr, entity_label_map)
    # print(normed_expr)
    # print(denormed_expr)
    if 'time_macro' in denormed_expr:
        try:
            approx_expr = get_approx_s_expr(denormed_expr)
        except:
            return 'null', []
        try:
            additional_clause = get_time_macro_clause(denormed_expr)
            approx_sparql = lisp_to_sparql(approx_expr)
            approx_sparql_end = approx_sparql.rfind('}')
            cat_sqarql = approx_sparql[:approx_sparql_end] + additional_clause + approx_sparql[approx_sparql_end:]
            # print(approx_sparql)
            # print(additional_clause)
            # print(cat_sqarql)

            cat_result = execute_query(cat_sqarql)

            return denormed_expr, cat_result
        except:
            return 'null', []
    else:
        query_expr = denormed_expr.replace('( ', '(').replace(' )', ')')
        # return query_expr, []
        try:
            # print('parse', query_expr)
            sparql_query = lisp_to_sparql(query_expr)
            # print('sparql', sparql_query)
            denotation = execute_query(sparql_query)
        except:
            query_expr = 'null'
            denotation = []
        return query_expr, denotation


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
    generation_dataset = load_json(f'outputs/webqsp_{split}_gen.json')

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

    with open(f'misc/webqsp_{split}_final_results.txt', 'w') as f:
        f.writelines([x+'\n' for x in lines])

def force_top_1_eval(split, predict_file):
    predictions = load_json(predict_file)
    generation_dataset = load_json(f'outputs/webqsp_{split}_gen.json')

    lines = []

    for feat in tqdm(generation_dataset, total=len(generation_dataset)):
        qid = feat['qid']
        pred = predictions[qid]

        entity_label_map = get_entity_mapping_from_top_candidates(feat)

        top_pred = pred[0]
        lf, answers = execute_normed_s_expr(top_pred, entity_label_map)

        lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer': answers,}))


    with open(f'misc/webqsp_{split}_gen_top_results.txt', 'w') as f:
        f.writelines([x+'\n' for x in lines])

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--pred_file', default=None, help='topk prediction file')
    args = parser.parse_args()
    if args.pred_file is None:
        args.pred_file = f'misc/webqsp_{args.split}_topk_generations.json'

    print('split', args.split, 'topk_file', args.pred_file)
    return args

if __name__=='__main__':
    args = _parse_args()
    # top_k_eval(args.split, args.pred_file)
    # force_top_1_eval(args.split, args.pred_file)
    # top_k_upperbound(args.split, args.pred_file)
    aggressive_top_k_eval(args.split, args.pred_file)