"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from json import load
from os import link
import sys
sys.path.append('.')
import json
from tqdm import tqdm
from collections import OrderedDict
import pickle
from itertools import chain

# from entity_linker.bert_entity_linker import BertEntityLinker
# from entity_linker.BERT_NER.bert import Ner
# from entity_linker import surface_index_memory
# from entity_linker.aaqu_entity_linker import IdentifiedEntity
# from entity_linker.aaqu_util import normalize_entity_name, remove_prefixes_from_name, remove_suffixes_from_name

from components.utils import *


def to_output_data_format(identified_entity):
    data = {}
    data['label'] = identified_entity.name
    data['mention'] = identified_entity.mention
    data['pop_score'] = identified_entity.score
    data['surface_score'] = identified_entity.surface_score
    data['id'] = identified_entity.entity.id
    data['aliases'] = identified_entity.entity.aliases
    data['perfect_match'] = identified_entity.perfect_match
    return data


def get_all_entity_candidates(linker, utterance):
    mentions = linker.get_mentions(utterance)
    identified_entities = []
    mids = set()
    # print(mentions)
    all_entities = []
    for mention in mentions:
        results_per_mention = []
        # use facc1
        entities = linker.surface_index.get_entities_for_surface(mention)
        # use google kg api
        # entities = get_entity_from_surface(mention)
        # if len(entities) == 0:
        #     entities = get_entity_from_surface(mention)
        # print('A Mention', mention)
        # print('Init Surface Entitis', len(entities), entities)
        if len(entities) == 0 and len(mention) > 3 and mention.split()[0] == 'the':
            mention = mention[3:].strip()
            entities = linker.surface_index.get_entities_for_surface(mention)
            # print('Removing the then Surface Entitis', len(entities), entities)
        elif len(entities) == 0 and f'the {mention}' in utterance:
            mention = f'the {mention}'
            entities = linker.surface_index.get_entities_for_surface(mention)
            # print('Adding the then Surface Entitis', len(entities), entities)

        if len(entities) == 0:
            continue

        entities = sorted(entities, key=lambda x:x[1], reverse=True)
        for i, (e, surface_score) in enumerate(entities):
            if e.id in mids:
                continue
            # Ignore entities with low surface score. But if even the top 1 entity is lower than the threshold,
            # we keep it
            perfect_match = False
            # Check if the main name of the entity exactly matches the text.
            # I only use the label as surface, so the perfect match is always True
            if linker._text_matches_main_name(e, mention):
                perfect_match = True
            ie = IdentifiedEntity(mention,
                                    e.name, e, e.score, surface_score,
                                    perfect_match)
            # self.boost_entity_score(ie)
            # identified_entities.append(ie)
            mids.add(e.id)
            results_per_mention.append(to_output_data_format(ie))
        results_per_mention.sort(key=lambda x: x['surface_score'], reverse=True)
        # print(results_per_mention[:5])
        all_entities.append(results_per_mention)

    return all_entities

def read_entity_results(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    records = {}
    for l in lines:
        # Question ID, Entity Mention, Start Position (of the normalized question), Length, Linked Entity ID, Entity Name, Entity Linking Score
        qid, mention, start, length, mid, name, score = tuple(l.split('\t'))
        r = {
            'qid': qid,
            'mention': mention, 
            'start': int(start),
            'length': int(length),
            'mid': 'm.' + mid[3:],
            'label': name,
            'score': float(score)
        }
        # records[qid]
        if qid in records:
            records[qid].append(r)
        else:
            records[qid] = [r]

    return records

def extract_entities(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    entitiy_tokens = []
    for t in toks:
        # normalize entity
        if t.startswith('m.'):
            entitiy_tokens.append(t)
    return entitiy_tokens

##  [--------- for stagg results ---------------------------////
def prune_extracted_entities(result):
    result = [x for x in result if x['score'] > 2]
    return result

def entity_coverage(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    linking_result = read_entity_results(f'stagg/webquestions.examples.{split}.e2e.top10.filter.tsv')

    covered = 0
    equal = 0
    counted = 0
    for i, data in enumerate(dataset):
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        counted += 1
        gt_s_expr = [parse['SExpr'] for parse in data['Parses']]
        gt_s_expr = [x for x in gt_s_expr if x != 'null']
        if not gt_s_expr:
            continue
        gt_entities_sets = [set(extract_entities(x)) for x in gt_s_expr]
        # exit()
        if data['QuestionId'] not in linking_result:
            continue
        extracted_set = linking_result[data['QuestionId']]
        extracted_set = prune_extracted_entities(extracted_set)
        extracted_entities = set([x['mid'] for x in extracted_set])

        
        is_covered = any([x.issubset(extracted_entities) for x in gt_entities_sets]) 
        covered += is_covered
        is_equal =  any([x == extracted_entities for x in gt_entities_sets])
        equal += is_equal
        # if is_covered and (not is_equal):
        #     print('----------------------')
        #     print('QUESTION', data['RawQuestion'])
        #     print('EXTRACT', extracted_set)
        #     print('DETECTED', extracted_entities)
        #     print('GT', gt_entities_sets)
        #     print(gt_s_expr)
    # print(dismatch_cnt)        
    print(covered/counted, equal / counted, len(dataset))
##  /////--------- for stagg results ---------------------------]



##  [--------- for bert ner ---------------------------////
def dump_entity_linking_for_training(split, keep=10):
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
        "entity_linker/data/freebase_complete_all_mention")
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:1")

    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')    
    el_results = OrderedDict()
    for ex in tqdm(dataset, total=len(dataset)):
        query = ex['RawQuestion']
        qid = str(ex['QuestionId'])
        all_candidates = get_all_entity_candidates(entity_linker, query)
        all_candidates = [x[:keep] for x in all_candidates]
        el_results[qid] = all_candidates

    with open(f'stagg/webqsp_{split}-entities.json', 'w') as f:
        json.dump(el_results, f)

def entity_coverage_with_bert_ner(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    linking_result = load_json(f'stagg/webqsp_{split}-entities.json')


    counted = 0
    all_first_covered = []
    topk_choices = [1,3,5,10]
    for i, data in enumerate(dataset):
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        counted += 1
        gt_s_expr = [parse['SExpr'] for parse in data['Parses']]
        gt_s_expr = [x for x in gt_s_expr if x != 'null']
        if not gt_s_expr:
            continue
        gt_entities_sets = [set(extract_entities(x)) for x in gt_s_expr]
        # exit()
        first_coverd = 1000000000
        lnk_result = linking_result[data['QuestionId']]
        for k in topk_choices:
            topk_set = set( chain(*[[x['id'] for x in entities_per_mention[:k]] for entities_per_mention in lnk_result]))
            if any([x.issubset(topk_set) for x in gt_entities_sets]):
                first_coverd = k
                break
        all_first_covered.append(first_coverd)
    print('Coverage Table')
    for k in topk_choices:
        print(k, sum([x <= k for x in all_first_covered])/counted)
##  /////--------- for bert ner ---------------------------]


##  [--------- for elq ---------------------------////
def fpr_evaluate(pred, answers):
    max_res = (0,0,0)
    for ans in answers:
        hit = len(pred & ans)
        if hit == 0:
            res = 0,0,0
        else:
            p = hit / len(pred)
            r = hit / len(ans)
            f = 2 * (p * r) / ( p + r)
            res = (f, p, r)
        if res[0] > max_res[0]:
            max_res = res
    return max_res
    
def entity_coverage_with_elq(split):
    threshold = -5
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    linking_result = load_json('misc/webqsp_{}_elq{}_mid.json'.format(split, threshold))
    linking_result = dict([(x['id'], x) for x in linking_result])

    counted = 0
    covered_cnt = 0
    equal_cnt = 0
    parsable_cnt = 0
    fpr_values = []
    for i, data in enumerate(dataset):
        skip = True
        for pidx in range(0,len(data["Parses"])):
            np = data["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(data["Parses"])==0 or skip):
            continue
        counted += 1
        gt_s_expr = [parse['SExpr'] for parse in data['Parses']]
        gt_s_expr = [x for x in gt_s_expr if x != 'null']
        if not gt_s_expr:
            continue
        parsable_cnt += 1
        gt_entities_sets = [set(extract_entities(x)) for x in gt_s_expr]
        lnk_result = linking_result[data['QuestionId']]
        extracted_entities = set(lnk_result['freebase_ids'])
        is_covered = any([x.issubset(extracted_entities) for x in gt_entities_sets]) 
        covered_cnt += is_covered
        is_equal =  any([x == extracted_entities for x in gt_entities_sets])
        equal_cnt += is_equal
        fpr_values.append(fpr_evaluate(extracted_entities, gt_entities_sets))

    print(parsable_cnt/counted, parsable_cnt, counted)
    print(covered_cnt/parsable_cnt, equal_cnt / parsable_cnt, len(dataset))
    print(covered_cnt/counted, equal_cnt / counted, len(dataset))
    agg_f1 = sum([x[0] for x in fpr_values])
    agg_pre = sum([x[1] for x in fpr_values])
    agg_rec = sum([x[2] for x in fpr_values])
    print('F1', agg_f1 / parsable_cnt, agg_f1 / counted)
    print('Pre', agg_pre / parsable_cnt, agg_pre / counted)
    print('Rec', agg_rec / parsable_cnt, agg_rec / counted)
##  /////--------- for elq ---------------------------]

if __name__=='__main__':
    # dump_entity_linking_for_training('train')
    # dump_entity_linking_for_training('test')
    # entity_coverage_with_bert_ner('test')
    entity_coverage_with_elq('test')
