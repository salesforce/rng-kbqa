"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import json
from tqdm import tqdm
from collections import OrderedDict
import pickle
from itertools import chain
import argparse
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api


import elq.main_dense as main_dense
from components.utils import *

def dump_entity_linking_for_training(split, threshold=-1.5):
    models_path = "models/" # the path where you stored the ELQ models
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    data_to_feed = []
    for ex in tqdm(dataset, total=len(dataset)):
        query = ex['RawQuestion']
        data_to_feed.append({'id': ex['QuestionId'], 'text': query})
    
    config = {
        "interactive": False,
        "biencoder_model": models_path+"elq_webqsp_large.bin",
        "biencoder_config": models_path+"elq_large_params.txt",
        "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "output_path": "logs/", # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path+"faiss_hnsw_index.pkl",
        "num_cand_mentions": 10,
        "num_cand_entities": 10,
        "threshold_type": "joint",
        "threshold": threshold,
    }

    args = argparse.Namespace(**config)
    models = main_dense.load_models(args, logger=None)
    predictions = main_dense.run(args, None, *models, test_data=data_to_feed)
    associate_with_kb_id(predictions)
    output_name = 'outputs/webqsp_{}_elq{}.json'.format(split, threshold)
    dump_json(predictions, output_name)

def associate_with_kb_id(predictions):
    id2wikidata = json.load(open("models/id2wikidata.json"))
    for pred in predictions:
        del pred['tokens']
        wiki_ids = []
        for tri in pred['pred_triples']:
            targetqid = id2wikidata.get(tri[0], 'null')
            wiki_ids.append(targetqid)
        pred['wiki_ids'] = wiki_ids


def wiki_id_to_freebase_id(wid):
    assert wid[0] == 'Q'
    qdict = get_entity_dict_from_api(wid)
    if 'P646' not in qdict['claims']:
        print(wid, 'NOT FOUND')
        return 'none'

    p646 = qdict['claims']['P646']
    # if len(p646) != 1:
    #     print('LEN > 1', p646)
    #     raise RuntimeError('More than one mapping')
    try:
        p646 = p646[0]
        mid = p646['mainsnak']['datavalue']['value']
        assert mid.startswith('/m/')
        mid = 'm.' + mid[3:]
        return mid
    except:
        return 'none'

def assign_freebase_id_to_results(split, threshold=-1.5):
    infile_name = 'outputs/webqsp_{}_elq{}.json'.format(split, threshold)
    predictions = load_json(infile_name)
    for pred in tqdm(predictions, total=len(predictions)):
        fids = []
        for wid in pred['wiki_ids']:
            if wid == 'null':
                fid = 'null'
            else:
                fid = wiki_id_to_freebase_id(wid)
            fids.append(fid)
        # print(fids)
        pred['freebase_ids'] = fids
        if 'none' in fids:
            print(fids)
            print(pred.keys())
    dump_json(predictions, 'outputs/webqsp_{}_elq{}_mid.json'.format(split, threshold))

if __name__=='__main__':
    dump_entity_linking_for_training('test', -5)
    assign_freebase_id_to_results('test', -5)
