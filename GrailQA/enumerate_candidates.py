"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import multiprocessing
import os
from typing import OrderedDict
import torch
from os.path import join
import sys
import argparse

from multiprocessing import Pool
from functools import partial

from components.utils import *
from executor.cached_enumeration import (
    CacheBackend,
    OntologyInfo,
    grail_enum_one_hop_one_entity_candidates,
    grail_enum_two_hop_one_entity_candidates,
    generate_all_logical_forms_for_literal,
    grail_enum_two_entity_candidates,
    grail_canonicalize_expr,
)
# get entity ambiguation prediction
from entity_linking.value_extractor import GrailQA_Value_Extractor
from components.disamb_dataset import (
    read_disamb_instances_from_entity_candidates,
)
from components.expr_parser import extract_entities, extract_relations

from grail_evaluate import process_ontology, SemanticMatcher
from nltk.tokenize import word_tokenize


MP_POOL_SIZE = 5
# True: use master property, False: use reserve property, None: do nothing
USE_MASTER_CONFIG = None

def _process_query(query):
    tokens = word_tokenize(query)
    proc_query = ' '.join(tokens).replace('``', '"').replace("''", '"')
    return proc_query

cnt = 0
def arrange_disamb_results_in_lagacy_format(split_id, entity_predictions_file):
    dataset_id = 'grail'
    example_cache = join('feature_cache', f'{dataset_id}_{split_id}_disamb_examples.bin')
    entities_file = f'outputs/grail_{split_id}_entities.json'
    if os.path.exists(example_cache):
        instances = torch.load(example_cache)
    else:
        dataset_file = join('outputs', f'grailqa_v1.0_{split_id}.json')
        instances = read_disamb_instances_from_entity_candidates(dataset_file, entities_file)
        torch.save(instances, example_cache)

    # build result index
    indexed_pred = load_json(entity_predictions_file)
    # for (feat, pred) in zip(valid_features, predicted_indexes):

    el_results = OrderedDict()
    for inst in instances:
        inst_result = {}
        normed_query = _process_query(inst.query)
        inst_result['question'] = normed_query
        pred_entities = OrderedDict()
        for problem in inst.disamb_problems:
            if len(problem.candidates) == 0:
                continue
            if len(problem.candidates) == 1 or problem.pid not in indexed_pred:
                pred_idx = 0
            else:
                # print('using predicted entity linking')
                pred_idx = indexed_pred[problem.pid]

            entity = problem.candidates[pred_idx]
            start_pos = normed_query.find(problem.mention) # -1 or strat
            pred_entities[entity.id] = {
                "mention": problem.mention,
                "label": entity.label,
                "friendly_name": entity.facc_label,
                "start": start_pos,
            }
        inst_result['entities'] = pred_entities
        el_results[inst.qid] = inst_result
            # pred_entities.append(problem.candidates[pred_idx].id)

    return el_results

def enumerate_candidates_from_entities_and_literals(entities, literals, use_master):
    logical_forms = []
    # print(entities)
    # print(literals)
    if len(entities) > 0:
            for entity in entities:
                logical_forms.extend(grail_enum_one_hop_one_entity_candidates(entity, use_master=use_master))
                lfs_2 = grail_enum_two_hop_one_entity_candidates(entity, use_master=use_master)
                logical_forms.extend(lfs_2)
    if len(entities) == 2:
        logical_forms.extend(grail_enum_two_entity_candidates(entities[0], entities[1], use_master=use_master))
    for literal in literals:
        logical_forms.extend(
            generate_all_logical_forms_for_literal(literal))

    return logical_forms

def process_single_item(item, el_results, extractor, use_master=True):
    item['qid'] = str(item['qid'])
    print(item['qid'])
    # no el results provided, using gt
    if el_results is None:
        entities = []
        entity_map = {}
        for node in item['graph_query']['nodes']:
            if node['node_type'] == 'entity':
                if node['id'] not in entities:
                    entities.append(node['id'])
                    entity_map[node['id']] = ' '.join(
                        node['friendly_name'].replace(";", ' ').split()[:5])
        literals = []
        for node in item['graph_query']['nodes']:
            if node['node_type'] == 'literal' and node['function'] not in ['argmin', 'argmax']:
                if node['id'] not in literals:
                    literals.append(node['id'])
        if len(entities) > 1:
            normed_query = _process_query(item['question'])
            entities = sorted(entities, key=lambda k: normed_query.find(entity_map[k].lower()))
    # using el results, for testing    
    else:
        # find entity linking
        entity_map = el_results[item['qid']]['entities']
        entities = sorted(set(entity_map.keys()), key=lambda k: entity_map[k]["start"])

        # print("linked entities:", entities)
        # literals = set()
        mentions = extractor.detect_mentions(item['question'])
        mentions = [extractor.process_literal(m) for m in mentions]
        literals = []
        for m in mentions:
            if m not in literals:
                literals.append(m)
    if 's_expression' in item:
        canonical_expr = grail_canonicalize_expr(item['s_expression'], use_master=use_master)
        # make logical forms
        logical_forms = enumerate_candidates_from_entities_and_literals(entities, literals, use_master=use_master)

        # if len(logical_forms) == 0:
        #     continue
        return {'qid': item['qid'], 'canonical_expr': canonical_expr, 's_expression': item['s_expression'], 'candidates': logical_forms}
    else:
        canonical_expr = 'null'
        logical_forms = enumerate_candidates_from_entities_and_literals(entities, literals, use_master=use_master)
        return {'qid': item['qid'], 'canonical_expr': canonical_expr, 's_expression': 'null', 'candidates': logical_forms}


# generate candidates
def generate_candidate_file(dataset_file, el_results, is_parallel=False):
    extractor = GrailQA_Value_Extractor()

    # el_fn = "graphq_el.json" if _gq1 else "grailqa_el.json"
    file_contents = load_json(dataset_file)

    process_func = partial(process_single_item, el_results=el_results, extractor=extractor, use_master=USE_MASTER_CONFIG)
    if is_parallel:
        CacheBackend.multiprocessing_preload()
        candidates_info = []
        with Pool(MP_POOL_SIZE) as p:
            candidates_info = p.map(process_func, file_contents, chunksize=100)
            # candidates_info = p.map(process_func, file_contents)
    else:
        candidates_info = []
        for i, item in enumerate(file_contents):
            print(i)
            candidates_info.append(process_func(item))
    candidates_info = [x for x in candidates_info if len(x['candidates'])]
    candidate_numbers = [len(x['candidates']) for x in candidates_info]
    print('AVG candidates', sum(candidate_numbers) / len(candidate_numbers), 'MAX', max(candidate_numbers))
    is_str_covered = [x['s_expression'] in x['candidates'] for x in candidates_info]
    print('Str coverage of orig expr', sum(is_str_covered) / len(is_str_covered), len(is_str_covered))
    is_str_covered = [x['canonical_expr'] in x['candidates'] for x in candidates_info]
    print('Str coverage of canonical expr', sum(is_str_covered) / len(is_str_covered), len(is_str_covered))
    is_str_same = [x['canonical_expr'] == x['s_expression'] for x in candidates_info]
    print('Canonical expr same with Orig ', sum(is_str_same) / len(is_str_covered), len(is_str_covered))
    return candidates_info

def pick_closest_target_expr(gt_expr, alter_exprs):
    gt_relations = set(extract_relations(gt_expr))
    
    sort_keys = []
    for expr in alter_exprs:
        e_relations = set(extract_relations(expr))
        r_dist = -len(gt_relations & e_relations) * 1.0 / len(gt_relations | e_relations)
        len_dist = -abs(len(expr) - len(gt_expr))
        # first relation overlapping then length difference
        sort_keys.append((r_dist, len_dist))
    print(sort_keys)
    selected_idx = min(list(range(len(alter_exprs))), key=lambda x: sort_keys[x])
    return alter_exprs[selected_idx]

def augment_edit_distance(candidates_info):
    reverse_properties, relation_dr, relations, upper_types, types = process_ontology('ontology/fb_roles', 'ontology/fb_types', 'ontology/reverse_properties')
    matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)
    hit_chance = 0
    ex_chance = 0
    count = 0
    augmented_lists = []
    for i, instance in enumerate(candidates_info):
        candidates = instance['candidates']
        gt = instance['canonical_expr']
        print(i, len(candidates))
        aux_candidates = []
        for c in candidates:
            if gt == 'null':
                ex = False
            else:
                ex = matcher.same_logical_form(gt, c)
            # tokens = []
            aux_candidates.append({'logical_form': c, 'ex': ex,})
        is_covered = any([x['ex'] for x in aux_candidates])
        hit_chance += is_covered
        is_exact = any([x['logical_form'] == gt for x in aux_candidates])
        ex_chance += is_exact

        if is_covered and not is_exact:
            # use relation overlapping to select the set with the
            alter_targets = [x['logical_form'] for x in aux_candidates if x['ex']]
            if len(alter_targets) == 1:
                target_expr = alter_targets[0]
            else:
               
                # exit()
                selected = pick_closest_target_expr(gt, alter_targets)
                target_expr = selected
        else:
            target_expr = gt

        instance['candidates'] = aux_candidates
        instance['target_expr'] = target_expr
        count += 1
        augmented_lists.append(instance)
    print('Coverage', hit_chance, count, hit_chance / count)
    print('Exact', ex_chance, count, ex_chance / count)
    return augmented_lists

# run evaluation
def sanity_check_proced_question():
        # el_fn = "graphq_el.json" if _gq1 else "grailqa_el.json"
    legacy_results = load_json("entity_linking/grailqa_el.json")
    new_results = load_json('tmp/tmp_el_results.json')
    count = 0
    for qid, leg_res in legacy_results.items():
        if qid not in new_results:
            continue
        new_res = new_results[str(qid)]
        if leg_res['question'] != new_res['question']:
            print('------------------------')
            print(new_res['question'])
            print(leg_res['question'])
            count += 1
    print(count)

def enumerate_candidates_for_ranking(split, pred_file):
    if split == 'train':
        el_results = None
    else:
        el_results = arrange_disamb_results_in_lagacy_format(split, pred_file)
        # temporalily save in case sth wrong
        # dump_json(el_results, f'tmp/tmp_el_results_{split}.json', indent=2)

    dataset_file = join('outputs', f'grailqa_v1.0_{split}.json')

    CacheBackend.init_cache_backend('grail')
    OntologyInfo.init_ontology_info('grail')
    candidates_info = generate_candidate_file(dataset_file, el_results)
    CacheBackend.exit_cache_backend()

    dump_json(candidates_info, f'misc/grail_{split}_candidates_intermidiate.json')
    # candidates_info = load_json(f'misc/grail_{split}_candidates_intermidiate.json')
    augmented_lists = augment_edit_distance(candidates_info)
    with open(f'outputs/grail_{split}_candidates-ranking.jsonline', 'w') as f:
        for info in augmented_lists:
            f.write(json.dumps(info) + '\n')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--pred_file', default=None, help='prediction file')
    args = parser.parse_args()
    if args.split != 'train':
        if args.pred_file is None:
            raise RuntimeError('A prediction file is required for evaluation and prediction (when split is not Train)')

    print('split', args.split, 'prediction', args.pred_file)
    return args

if __name__ == '__main__':
    # assert len(sys.argv) >= 2
    args = _parse_args()  
    enumerate_candidates_for_ranking(args.split, args.pred_file)
