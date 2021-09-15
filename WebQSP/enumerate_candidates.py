"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse

from collections import Counter, OrderedDict
from components.utils import *
from components.expr_parser import extract_entities, parse_s_expr, tokenize_s_expr
# from executor.search_over_graphs import generate_all_logical_forms_alpha, generate_all_logical_forms_2, \
    # get_vocab_info_online, generate_all_logical_forms_for_literal
# get entity ambiguation prediction
# from components_bak.candidate_enumerator import CacheBackend, generate_all_logical_forms_alpha, generate_all_logical_forms_2, generate_logical_forms_for_two_entities
from executor.cached_enumeration import CacheBackend, OntologyInfo, webqsp_enum_one_hop_one_entity_candidates, webqsp_enum_two_hop_one_entity_candidates, webqsp_enum_two_entity_candidates
from executor.logic_form_util import same_logical_form

def get_logical_forms_from_entities(entities):
    logical_forms = []
    if not entities:
        return []

    for entity in entities:
        logical_forms.extend(webqsp_enum_one_hop_one_entity_candidates(entity))
        # print(len(logical_forms))
        lfs_2 = webqsp_enum_two_hop_one_entity_candidates(entity)
        logical_forms.extend(lfs_2)
    if len(entities) == 2:
        logical_forms.extend(webqsp_enum_two_entity_candidates(entities[0], entities[1]))
    return logical_forms


def assign_ex_label_to_logical_forms(logical_forms, approx_exprs, is_training=True):
    # 'logical_form': c, 'ex': ex, 'edit_distance': edit_dist})
    ret_logical_forms = []
    for lf in logical_forms:
        if is_training:
            ex = False
            for ap_e in approx_exprs:
                if same_logical_form(lf, ap_e):
                    ex = True
                    break
            ret_logical_forms.append({'logical_form': lf , 'ex': ex})
        else:
            ex = False
            ret_logical_forms.append({'logical_form': lf , 'ex': ex})
    # ret_logical_forms
    return ret_logical_forms


def covered_relation(r):
    if r.startswith('common.') or r.startswith('type.') or r.startswith('kg.') or r.startswith('user.'):
        return False
    return True

def select_target_expr_for_training(approx_s_exprs, is_entity_covered=[]):
    # dev
    gt_entities_sets = [set(extract_entities(x)) for x in approx_s_exprs]
    if not is_entity_covered:
        is_entity_covered = [True] * len(gt_entities_sets)
    tokenized_expr = []
    is_relation_covered = []
    for s_expr in approx_s_exprs:
        toks = tokenize_s_expr(s_expr)
        tokenized_expr.append(toks)
        relations = [x for x in toks if '.' in x and not x.startswith('m.') and '^^http' not in x]
        is_relation_covered.append(all([covered_relation(r) for r in relations]))

    is_expr_covered = [a and b for (a,b) in zip(is_entity_covered, is_relation_covered)]
    # if sum(is_expr_covered) != sum(is_entity_covered):
    #     print(approx_s_exprs)
        # exit()
    if any(is_expr_covered):
        simplest_idx = sorted([i for i in range(len(approx_s_exprs)) if is_expr_covered[i]],
                    key=lambda x: (len(gt_entities_sets[x]), len(tokenized_expr[x])))[0]
    else:
        simplest_idx = sorted([i for i in range(len(approx_s_exprs)) if is_entity_covered[i]],
                    key=lambda x: (len(gt_entities_sets[x]), len(tokenized_expr[x])))[0]
    return simplest_idx

def ordered_set_as_list(xs):
    ys = []
    for x in xs:
        if x not in ys:
            ys.append(x)
    return ys

# use elq
# generate candidates
def generate_candidate_file(split, use_gt_entities=False, lnk_split=None):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    if lnk_split is None:
        lnk_split = split
    linking_result = load_json('misc/webqsp_{}_elq-5_mid.json'.format(lnk_split))
    linking_result = dict([(x['id'], x) for x in linking_result])

    candidates_info = []
    covered = 0
    entity_covered = 0
    target_covered = 0
    required_entity_num = []
    length_of_candidates = []

    for i, data in enumerate(dataset):
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        qid = data['QuestionId']

        gt_s_exprs = [parse['SExpr'] for parse in data['Parses']]
        gt_s_exprs = [x for x in gt_s_exprs if x != 'null']
        approx_s_exprs = [get_approx_s_expr(x) for x in gt_s_exprs]
        gt_entities_sets = [ordered_set_as_list(extract_entities(x)) for x in gt_s_exprs]

        if use_gt_entities:
            # pick the target
            # approx_s_exprs = 
            if not gt_entities_sets:
                target_s_expr = 'null'
                target_full_s_expr = 'null'
                entities = []
            else:
                simplest_idx = select_target_expr_for_training(approx_s_exprs)
                target_s_expr = approx_s_exprs[simplest_idx]
                target_full_s_expr = gt_s_exprs[simplest_idx]
                entities = gt_entities_sets[simplest_idx]
                # if len(gt_entities_sets) > 1:
                #     print('----USE GT----')
                #     print(approx_s_exprs)
                #     print(target_s_expr)
        else:
            query = data['RawQuestion'].lower()
            lnk_result = linking_result[data['QuestionId']]
            entities = lnk_result['freebase_ids']
            entity_mention_str_mapping = dict([ (e, tuple_str[1]) for (e, tuple_str) in zip(entities, lnk_result['pred_tuples_string'])])
            entities = ordered_set_as_list(entities)
            # some entities  are none because elq fails to be linked with freebase eid
            entities = [x for x in entities if x != 'none']
            entities.sort(key=lambda x: query.find(entity_mention_str_mapping[x]))
            # pick the target
            # covered and x
            is_entity_covered = [set(x).issubset(entities) for x in gt_entities_sets]
            # sorted([ for i in range(len(approx_s_exprs))])
            if not any(is_entity_covered):
                target_s_expr = 'null'
                target_full_s_expr = 'null'
            else:
                simplest_idx = select_target_expr_for_training(approx_s_exprs, is_entity_covered)
                target_s_expr = approx_s_exprs[simplest_idx]
                target_full_s_expr = gt_s_exprs[simplest_idx]
                # if sum(is_entity_covered) > 1:
                #     print('--------')
                #     print(approx_s_exprs)
                #     print(target_s_expr)
        required_entity_num.append(len(entities))
        # make logical forms
        logical_forms = get_logical_forms_from_entities(entities)
        length_of_candidates.append(len(logical_forms))
        if len(gt_s_exprs) and any([x in logical_forms for x in gt_s_exprs]):
            covered += 1
        if any([set(x).issubset(entities) for x in gt_entities_sets]):
            entity_covered += 1
        if target_s_expr in logical_forms:
            target_covered += 1
        print(i, len(dataset), len(logical_forms))
        logical_forms = assign_ex_label_to_logical_forms(logical_forms, approx_s_exprs, is_training='train' in split)
        # if target_s_expr != target_full_s_expr:
        #     print(target_s_expr)
        #     print(target_full_s_expr)
        candidates_info.append({'qid': qid, 'target_s_expr': target_s_expr, 'target_full_s_expr': target_full_s_expr, 'gt_s_expressions':gt_s_exprs , 'approx_s_expressions': approx_s_exprs, 'candidates': logical_forms})
    required_entity_num = Counter(required_entity_num)
    print('Expr Cov', covered, len(candidates_info), covered/len(candidates_info))
    print('Approx Cov', target_covered, len(candidates_info), target_covered/len(candidates_info))
    print('Entity Cov', entity_covered, len(candidates_info), entity_covered/len(candidates_info))
    print('Num Detected', required_entity_num)
    print('Avg Num', sum(length_of_candidates) / len(length_of_candidates))

    dump_json(candidates_info, f'outputs/webqsp_{split}_candidates-ranking.json')

def approx_time_macro_ast(node):
    # print('NODE', node.construction, node.logical_form())
    if (node.construction == 'AND' and
        node.fields[0].construction == 'JOIN' and
        node.fields[0].fields[0].construction == 'SCHEMA' and 
        'time_macro' in node.fields[0].fields[0].val):
        return node.fields[1]
    else:
        new_fileds = [approx_time_macro_ast(x) for x in node.fields]
        node.fields = new_fileds
        return node

def get_approx_s_expr(x):
    if not ('time_macro' in x):
        return x

    ast = parse_s_expr(x)
    approx_ast = approx_time_macro_ast(ast)
    approx_x = approx_ast.compact_logical_form()
    return approx_x

def dump_candidates_file_for_training():
    CacheBackend.init_cache_backend('webqsp')
    OntologyInfo.init_ontology_info()
    generate_candidate_file('ptrain', use_gt_entities=True, lnk_split='train')
    generate_candidate_file('pdev', use_gt_entities=False, lnk_split='train')
    CacheBackend.exit_cache_backend()


def dump_candidates_file_for_split(split):
    CacheBackend.init_cache_backend('webqsp')
    OntologyInfo.init_ontology_info()
    if split == 'test':
        generate_candidate_file('test')
    elif split == 'pdev':
        generate_candidate_file('pdev', use_gt_entities=False, lnk_split='train')
    elif split == 'ptrain':
        generate_candidate_file('ptrain', use_gt_entities=True, lnk_split='train')
    elif split == 'train':
        generate_candidate_file('train', use_gt_entities=True)
    else:
        raise RuntimeError('invalid split')
    CacheBackend.exit_cache_backend()

# def dump_partial_candidates_file():
#     CacheBackend.init_cache_backend('webqsp')
#     OntologyInfo.init_ontology_info()
#     generate_candidate_file('ptrain', use_gt_entities=True, lnk_split='train')
#     generate_candidate_file('pdev', use_gt_entities=False, lnk_split='train')
#     CacheBackend.exit_cache_backend()

def inspect_coverage_breakdown(split):

    threshold = -5
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    linking_result = load_json('misc/webqsp_{}_elq{}_mid.json'.format(split, threshold))
    linking_result = dict([(x['id'], x) for x in linking_result])

    candidates_info = load_json(f'outputs/webqsp_{split}_candidates.json')
    # candidates_info = load_json(f'outputs/webqsp_{split}_candidates-ranking.json')

    covered = 0
    entity_covered = 0
    coverage_break_down = []
    length_of_candidates = []
    for i, (data, candidates) in enumerate(zip(dataset, candidates_info)):
        logical_forms = [x['logical_form'] for x in candidates['candidates']]
        length_of_candidates.append(len(logical_forms))
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        qid = data['QuestionId']

        gt_s_expr = [parse['SExpr'] for parse in data['Parses']]
        gt_s_expr = [x for x in gt_s_expr if x != 'null']
        approx_s_expr = candidates['approx_s_expressions']

        lnk_result = linking_result[data['QuestionId']]
        entities = set(lnk_result['freebase_ids'])

        is_expr_covered = len(approx_s_expr) and any([x in logical_forms for x in approx_s_expr])
        # print(gt_s_expr, is_expr_covered)
        # exit()
        covered += is_expr_covered
        gt_entities_sets = [set(extract_entities(x)) for x in approx_s_expr]
        is_entity_covered = len(gt_entities_sets) and any([x.issubset(entities) for x in gt_entities_sets])
        entity_covered += is_entity_covered

        minimum_required = min([len(x) for x in gt_entities_sets]) if gt_entities_sets else 0
        coverage_break_down.append({'min': minimum_required, 'detected': len(entities),
            'entity_covered': is_entity_covered, 'expr_covered': is_expr_covered})
        
        # if minimum_required == 1 and is_entity_covered and not is_expr_covered:
        #     if not any(['ARG' in x or 'time_macro' in x for x in approx_s_expr]):
        #         print('----------1 Entity Not Covered---------')
        #         print(data['RawQuestion'])
        #         print(entities)
        #         print(approx_s_expr)

    # length 1 covered
    length1_breakdown = [x for x in coverage_break_down if x['min'] == 1 and x['entity_covered']]
    print('One entity', sum([x['expr_covered'] for x in length1_breakdown] ) / len(length1_breakdown), len(length1_breakdown))

    length2_breakdown = [x for x in coverage_break_down if x['min'] == 2 and x['entity_covered']]
    print('Two Entities', sum([x['expr_covered'] for x in length2_breakdown]) / len(length2_breakdown), len(length2_breakdown))

    print(covered, len(candidates_info), covered/len(candidates_info))
    print(entity_covered, len(candidates_info), entity_covered/len(candidates_info))
    print(sum(length_of_candidates) / len(length_of_candidates))


def inspect_uncovered_skeleton(split):
    threshold = -5
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    linking_result = load_json('outputs/webqsp_{}_elq{}_mid.json'.format(split, threshold))
    linking_result = dict([(x['id'], x) for x in linking_result])

    candidates_info = load_json(f'tmp/tmp_{split}_candidates.json')

    covered = 0
    entity_covered = 0
    coverage_break_down = []
    length_of_candidates = []
    skeletons = set()

    two_entity_detected = 0
    two_entity_detected_but_one_required = 0
    for i, (data, candidates) in enumerate(zip(dataset, candidates_info)):
        logical_forms = candidates['candidates']
        length_of_candidates.append(len(logical_forms))
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        qid = data['QuestionId']

        gt_s_expr = [parse['SExpr'] for parse in data['Parses']]
        gt_s_expr = [x for x in gt_s_expr if x != 'null']

        lnk_result = linking_result[data['QuestionId']]
        entities = set(lnk_result['freebase_ids'])

        
        is_expr_covered = len(gt_s_expr) and any([x in logical_forms for x in gt_s_expr])
        covered += is_expr_covered
        gt_entities_sets = [set(extract_entities(x)) for x in gt_s_expr]
        is_entity_covered = len(gt_entities_sets) and any([x.issubset(entities) for x in gt_entities_sets])
        entity_covered += is_entity_covered

        minimum_required = min([len(x) for x in gt_entities_sets]) if gt_entities_sets else 0
        coverage_break_down.append({'min': minimum_required, 'detected': len(entities),
            'entity_covered': is_entity_covered, 'expr_covered': is_expr_covered})
        
        # if minimum_required == 2 and is_entity_covered and not is_expr_covered:
        #     print('----------2 Entity Not Covered---------')
        #     print(data['RawQuestion'])
        #     print(entities)
        #     target_s_expr = candidates['target_s_expr']
        #     print(target_s_expr)
        #     sk = parse_s_expr(target_s_expr).skeleton_form()
        #     print(sk)
        #     skeletons.add(sk)
        if minimum_required == 2 and candidates['target_s_expr'] != 'null':
            print('----------2 Entity Not Covered---------')
            print(data['RawQuestion'])
            # print(entities)
            target_s_expr = candidates['target_s_expr']
            print(target_s_expr)
            sk = parse_s_expr(target_s_expr).skeleton_form()
            print(sk)
            skeletons.add(sk)        
        # if len(entities) == 2 and is_entity_covered:
        #     two_entity_detected += 1
        #     if minimum_required == 1:
        #         two_entity_detected_but_one_required += 1
    # length 1 covered
    length2_breakdown = [x for x in coverage_break_down if x['min'] == 2 and x['entity_covered']]
    print('Two Entities', sum([x['expr_covered'] for x in length2_breakdown]) / len(length2_breakdown), len(length2_breakdown))

    for s in skeletons:
        print(s)
    
    # print('Two Detected', two_entity_detected, 'Two But One', two_entity_detected_but_one_required)

def inspect_target_difference():
    withbase_candidates_info = load_json(f'tmp/tmp_train_candidates.json')
    nobase_candidates_info = load_json(f'tmp/tmp_train_candidates_nobase.json')
    for with_info, no_info in zip(withbase_candidates_info, nobase_candidates_info):
        print('checking')
        if with_info['target_s_expr'] != no_info['target_s_expr']:
            print('----------------------')
            print(with_info['target_s_expr'])
            print(no_info['target_s_expr'])
            print(with_info['gt_s_expressions'])


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    args = parser.parse_args()

    print('split', args.split)
    return args

if __name__ == '__main__':
    # assert len(sys.argv) >= 2
    args = _parse_args()
    if args.split == 'traindev':
        dump_candidates_file_for_training()
    else:
        dump_candidates_file_for_split(args.split)
