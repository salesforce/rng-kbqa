from json import load
import sys
sys.path.append('.')
import json
from tqdm import tqdm
from collections import OrderedDict
import pickle
from itertools import chain
import argparse

from entity_linker.bert_entity_linker import BertEntityLinker
from entity_linker.BERT_NER.bert import Ner
from entity_linker import surface_index_memory
from entity_linker.aaqu_entity_linker import IdentifiedEntity
from entity_linker.aaqu_util import normalize_entity_name, remove_prefixes_from_name, remove_suffixes_from_name



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    return parser.parse_args()


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


def dump_entity_linking(split):
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
        "entity_linker/data/freebase_complete_all_mention")
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:1")
    get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")
    print
    # get_all_entity_candidates(entity_linker, "newton per metre is the unit of measurement for surface tension in which system of measurement?")

    datafile = f'data/grailqa_v1.0_{split}.json'
    with open(datafile) as f:
        data = json.load(f)
    
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        # print(ex.keys())
        query = ex['question']
        qid = str(ex['qid'])
        el_results[qid] = get_all_entity_candidates(entity_linker, query)

    with open(f'candidates/grail_{split}-entities.bin', 'wb') as f:
        pickle.dump(el_results, f)


def details_of_diff(rep, given):
    # print(given)
    print(['{} --> {}'.format(given['entities'][e]['mention'], e) for e in given['entities']])
    print('By Surface Score')
    for x in rep:
        if x:
            print(x[0]['mention'], '\t-->\t', [y['id'] for y in x[:5]])

    print('By Pop Score')
    for x in [sorted(x, key=lambda y: y['pop_score'], reverse=True) for x in rep]:
        if x:
            print(x[0]['mention'], '\t-->\t', [y['id'] for y in x[:5]])

    
def cross_check_entity_link(split):
    with open(f'candidates/grail_{split}-entities.bin', 'rb') as f:
        rep_results = pickle.load(f)
    with open('entity_linking/grailqa_el.json') as f:
        given_results = json.load(f)
    
    num_diff = 0
    for qid in rep_results:
        rep = rep_results[qid]
        given = given_results[qid]

        # compare top 1
        # [x.sort(key=lambda y: y['pop_score'], reverse=True) for x in rep]
        rep_set = set([x[0]['id'] for x in rep if len(x)])
        given_set = set(given['entities'].keys())
        if rep_set != given_set:
            num_diff += 1
            print('-------------DIFF', given['question'])
            # print('REP', rep_set)
            # print('GIVEN', given_set)
            details_of_diff(rep, given)
    
    print('TOTAL DIFF', num_diff / len(rep_results))


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

def check_entity_link_coverage(split):
    with open(f'data/grailqa_v1.0_{split}.json') as f:
        dataset = json.load(f)
    with open(f'candidates/grail_{split}-entities.bin', 'rb') as f:
        rep_results = pickle.load(f)
    
    all_first_covered = []
    all_mention_num = []
    topk_choices = [1,3,5,10,20,50,100,1000]
    for data in dataset:
        qid = str(data['qid'])
        rep = rep_results[qid]

        s_expr = data['s_expression']
        entities_in_gt = set(extract_entities(s_expr))
        # compare top 1
        # [x.sort(key=lambda y: y['pop_score'], reverse=True) for x in rep]
        rep_set = set([x[0]['id'] for x in rep if len(x)])
        # arbitary large
        first_coverd = 1000000000
        all_mention_num.extend([len(x) for x in rep])
        for k in topk_choices:
            topk_set = set( chain(*[[x['id'] for x in entities_per_mention[:k]] for entities_per_mention in rep]) )
            if entities_in_gt.issubset(topk_set):
                first_coverd = k
                break
        # print(first_coverd)
        all_first_covered.append(first_coverd)
    print('Max Linked', max(all_mention_num), 'Average Linked', sum(all_mention_num)/len(all_mention_num))
    print('Coverage Table')
    for k in topk_choices:
        print(k, sum([x <= k for x in all_first_covered])/len(all_first_covered))

def check_given_entity_link_coverage(split):
    with open(f'data/grailqa_v1.0_{split}.json') as f:
        dataset = json.load(f)
    with open('entity_linking/grailqa_el.json') as f:
        given_results = json.load(f)

    all_first_covered = []
    for data in dataset:
        qid = str(data['qid'])
        given = given_results[qid]


        s_expr = data['s_expression']
        entities_in_gt = set(extract_entities(s_expr))
        # compare top 1
        given_set = set(given['entities'].keys())
        # arbitary large
        all_first_covered.append(entities_in_gt.issubset(given_set))
    print('Cov Given', sum(all_first_covered)/len(all_first_covered))


def dump_entity_linking_for_training(split, keep=10):
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
        "entity_linker/data/freebase_complete_all_mention")
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/trained_ner_model/", device="cuda:0")
    sanity_checking = get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")
    print('RUNNING Sanity Checking on untterance')
    print('\t', "the music video stronger was directed by whom")
    print('Checking result', sanity_checking[0][:2])
    print('Checking result should successfully link stronger to some nodes in Freebase (MIDs)')
    print('If checking result does not look good please check if the linker has been set up successfully')

    datafile = f'outputs/grailqa_v1.0_{split}.json'
    with open(datafile) as f:
        data = json.load(f)
    
    el_results = {}
    for ex in tqdm(data, total=len(data)):
        # print(ex.keys())
        query = ex['question']
        qid = str(ex['qid'])
        all_candidates = get_all_entity_candidates(entity_linker, query)
        all_candidates = [x[:keep] for x in all_candidates]
        el_results[qid] = all_candidates

    with open(f'outputs/grail_{split}_entities.json', 'w') as f:
        json.dump(el_results, f)

if __name__=='__main__':
    args = _parse_args()
    dump_entity_linking_for_training(args.split)
