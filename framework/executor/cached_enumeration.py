"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""
 Copyright 2021, Ohio State University (Yu Gu)
 Yu Gu  <gu.826@osu.edu>
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

"""
We extend the origial Grail code by
* adding classes for managing KB Cache
  invloved classes: OntologyInfo, CacheBackend, FBCacheBase,
                    FBLinkedRelationCache, FBTwoHopPathCache, FBQueryCache
* adding cached enumeration methods for enumerating logical forms for both GrailQA and WebQAP
  involved functions: grail_*, and webqsp_* (all lines from line 388 and below)
"""

import json
from types import CodeType

import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from os.path import join
from os.path import exists
from components.utils import load_json, dump_json
from components.expr_parser import *
from executor.logic_form_util import none_function, count_function
from executor.sparql_executor import get_adjacent_relations, get_2hop_relations


from multiprocessing import Process, Manager


domain_info = defaultdict(lambda: 'base')
with open('ontology/domain_info', 'r') as f:
    # domain_info = json.load(f)
    domain_info.update(json.load(f))

with open('ontology/fb_roles', 'r') as f:
    contents = f.readlines()

with open('ontology/fb_types', 'r') as f:
    type_infos = f.readlines()

subclasses = defaultdict(lambda: set())
for line in type_infos:
    fields = line.split()
    subclasses[fields[2]].add(fields[0])
    subclasses[fields[2]].add(fields[2])

# subclasses = {k: v for k, v in sorted(subclasses.items(), key=lambda x: len(x[1]), reverse=True)}

domain_dict_relations = defaultdict(lambda: set())
domain_dict_types = defaultdict(lambda: set())

relations_info: Dict[str, Tuple] = {}  # stores domain and range information for all relations
date_relations = set()
numerical_relations = set()

for line in contents:
    fields = line.split()
    domain_dict_relations[domain_info[fields[1]]].add(fields[1])
    domain_dict_types[domain_info[fields[0]]].add(fields[0])
    domain_dict_types[domain_info[fields[2]]].add(fields[2])
    relations_info[fields[1]] = (fields[0], fields[2])
    if fields[2] in ['type.int', 'type.float']:
        numerical_relations.add(fields[1])
    elif fields[2] == 'type.datetime':
        date_relations.add(fields[1])

def load_reverse_property(inverse=False, dataset=None):
    data = load_json('ontology/full_reverse_properties.json')
    if inverse:
        data = dict([(v, k) for (k, v) in data.items()])
    # do not canonicalize to ilegal
    if dataset is not None:
        data =  dict([(k, v) for (k, v) in data.items() if legal_relation(v, dataset)])
    return data

class OntologyInfo:
    reverse_propert = None
    master_property = None
    @classmethod
    def init_ontology_info(cls, dataset=None):
        cls.reverse_property = load_reverse_property(inverse=False, dataset=dataset)
        cls.master_property = load_reverse_property(inverse=True, dataset=dataset)

class CacheBackend:
    cache = None
    @classmethod
    def init_cache_backend(cls, dataset):
        FBCacheBase.DATASET = dataset
        cls.cache = FBQueryCache()

    @classmethod
    def multiprocessing_preload(cls):
        cls.cache.multiprocessing_preload()

    @classmethod
    def exit_cache_backend(cls):
        cls.cache.save()
        cls.cache = None

class FBCacheBase:
    PREFIX = 'cache'
    FILENAME = 'base'
    DATASET = 'base'
    def __init__(self):
        self.ready = False
        self.update_count = 0
        self.data = {}

    def cache_filename(self):
        return join(self.PREFIX, '{}-{}'.format(self.DATASET, self.FILENAME))

    def load(self):
        fname = self.cache_filename()
        if exists(fname):
            print('Load relation cache from', fname)
            self.data = load_json(fname)
        else:
            self.data = {}
        self.ready = True

    def multiprocessing_preload(self):
        self.load()
        self.manager = Manager()
        self.data = self.manager.dict(self.data)

    def save(self):
        fname = self.cache_filename()
        print(self.update_count, len(self.data), type(self.data))
        if isinstance(self.data, dict):
            print('save relation cache to', fname)
            if self.update_count:
                dump_json(self.data, fname)
        else:
            print('save relation cache to', fname)
            # dump_json(dict(self.data), fname)
            dump_json(self.data.copy(), fname)

class FBLinkedRelationCache(FBCacheBase):
    FILENAME = 'LinkedRelation.bin'
    def query_in_out_relation(self, entity):
        if not self.ready:
            self.load()

        if entity in self.data:
            # print('HIT QUERY R', entity)
            return self.data[entity]
        # print('NOT HIT R', entity)
        in_r, out_r = get_adjacent_relations(entity)
        in_r, out_r = list(in_r), list(out_r)
        # in_r, out_r = self.dataset_specific_prune((in_r, out_r))
        self.update_count += 1
        self.data[entity] = (in_r, out_r)
        return (in_r, out_r)

    def dataset_specific_prune(self, relations):
        if self.DATASET == 'grail':
            in_r, out_r = relations
            in_r = [r for r in in_r if legal_relation(r, self.DATASET)]
            out_r = [r for r in out_r if legal_relation(r, self.DATASET)]
            return in_r, out_r
        else:
            return relations

class FBTwoHopPathCache(FBCacheBase):
    FILENAME = 'TwoHopPath.bin'
    def query_two_hop_paths(self, entity):
        if not self.ready:
            self.load()

        if entity in self.data:
            # print('HIT QUERY P', entity)
            return self.data[entity]
        # print('NOT HIT P', entity)
        paths = get_2hop_relations(entity)[2]
        paths = self.dataset_specific_prune(paths)
        self.update_count += 1
        self.data[entity] = paths
        return paths

    def dataset_specific_prune(self, two_hops):
        if self.DATASET == 'grail':
            two_hops = [(a, b) for (a, b) in two_hops if legal_relation(a, self.DATASET) and legal_relation(b, self.DATASET)]
            return two_hops
        else:
            return two_hops

class FBQueryCache:
    def __init__(self):
        self.linked_relation_cache = FBLinkedRelationCache()
        self.two_hop_paths_cahce = FBTwoHopPathCache()

    def query_relations(self, entity):
        return self.linked_relation_cache.query_in_out_relation(entity)

    def query_in_relation(self, entity):
        return self.linked_relation_cache.query_in_out_relation(entity)[0]

    def query_out_relation(self, entity):
        return self.linked_relation_cache.query_in_out_relation(entity)[1]

    def query_two_hop_paths(self, entity):
        return self.two_hop_paths_cahce.query_two_hop_paths(entity)

    def multiprocessing_preload(self):
        self.linked_relation_cache.multiprocessing_preload()
        self.two_hop_paths_cahce.multiprocessing_preload()

    def save(self):
        self.linked_relation_cache.save()
        self.two_hop_paths_cahce.save()


def add_node_to_G(G, node):
    G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
               function=node['function'])


def legal_relation(r, dataset='grail', force_in_bank=True):
    # print("gq1:", gq1)
    if r.endswith('#R'):
        r = r[:-2]
    if force_in_bank and (r not in relations_info):
        return False
    if r.startswith('common.') or r.startswith('type.') or r.startswith('kg.') or r.startswith('user.'):
        return False
    if dataset == 'grail':
        if r.startswith('base.') or r.startswith('dataworld') or r.startswith('freebase'):
            return False
    return True

def legal_domain(r, dataset='grail'):
    return legal_relation(r, dataset, False)

def generate_all_logical_forms_2(entity: str):
    if (CacheBackend.cache is not None):
        paths = CacheBackend.cache.query_two_hop_paths(entity)
    else:
        paths = get_2hop_relations(entity)[2]

    lfs = []
    for path in paths:
        if path[0][-2:] == "#R":
            if not legal_relation(path[0][:-2]):
                continue
            relation0 = '(R ' + path[0][:-2] + ')'
        else:
            if not legal_relation(path[0]):
                continue
            relation0 = path[0]
        if path[1][-2:] == "#R":
            if not legal_relation(path[1][:-2]):
                continue
            typ = relations_info[path[1][:-2]][1]
            relation1 = '(R ' + path[1][:-2] + ')'
        else:
            if not legal_relation(path[1]):
                continue
            typ = relations_info[path[1]][0]
            relation1 = path[1]
        lf = '(AND ' + typ + ' (JOIN ' + relation1 + ' (JOIN ' + relation0 + ' ' + entity + ')))'
        lfs.append(lf)
        lf = '(COUNT (AND ' + typ + ' (JOIN ' + relation1 + ' (JOIN ' + relation0 + ' ' + entity + '))))'
        lfs.append(lf)

        # G = build_graph_from_path(entity, path)
        # G1 = deepcopy(G)
        # lf = none_function(G, 2)
        # lfs.append(lf)
        # lf = count_function(G1, 2)
        # lfs.append(lf)
    return lfs

def generate_all_logical_forms_for_literal(value: str):
    lfs = []
    date = not (value.__contains__('integer') or value.__contains__('float'))
    if date:
        for r in date_relations:
            if legal_relation(r):
                lfs.append(f'(AND {relations_info[r][0]} (JOIN {r} {value}))')
    else:
        for r in numerical_relations:
            if legal_relation(r):
                lfs.append(f'(AND {relations_info[r][0]} (JOIN {r} {value}))')

    return lfs

# The alpha version only considers one entity and one step
def generate_all_logical_forms_alpha(entity: str,
                                     domains: List[str] = None,
                                     ):
    def r_in_domains(domains0, r0):
        for domain in domains0:
            if r0 in domain_dict_relations[domain]:
                return True

        return False

    if (CacheBackend.cache is not None):
        in_relations_e, out_relations_e = CacheBackend.cache.query_relations(entity)
    else:  # online executing the sparql query
        in_relations_e, out_relations_e = get_adjacent_relations(entity)

    lfs = []
    # if len(entities) == 0 or len(domains) == 0:
    #     return lfs
    if len(in_relations_e) > 0:
        for r in in_relations_e:
            if not legal_relation(r):
                continue
            if not domains or r_in_domains(domains, r):
                domain_r = relations_info[r][0]
                if len(subclasses[domain_r]) > 100:
                    subclasses[domain_r] = set()
                subclasses[domain_r].add(domain_r)
                for sub_domain_r in [domain_r]:
                    G = nx.MultiDiGraph()
                    node = {'nid': 0, 'id': entity, 'node_type': 'entity', 'question_node': 0,
                            'function': "none"}
                    G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                               function=node['function'])
                    node1 = {'nid': 1, 'id': sub_domain_r, 'node_type': 'class', 'question_node': 1,
                             'function': "none"}
                    G.add_node(node1['nid'], id=node1['id'], type=node1['node_type'],
                               question=node1['question_node'],
                               function=node1['function'])

                    G.add_edge(1, 0, relation=r, reverse=False, visited=False)
                    G.add_edge(0, 1, relation=r, reverse=True, visited=False)

                    G1 = deepcopy(G)

                    lf = none_function(G, 1)
                    lfs.append(lf)

                    lf = count_function(G1, 1)
                    lfs.append(lf)


    if len(out_relations_e) > 0:
        for r in out_relations_e:
            if not legal_relation(r):
                continue
            if not domains or r_in_domains(domains, r):
                range_r = relations_info[r][1]
                if len(subclasses[range_r]) > 100:
                    subclasses[range_r] = set()
                subclasses[range_r].add(range_r)
                # for sub_range_r in subclasses[range_r]:
                for sub_range_r in [range_r]:
                    G = nx.MultiDiGraph()
                    node = {'nid': 0, 'id': entity, 'node_type': 'entity', 'question_node': 0,
                            'function': "none"}
                    G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                               function=node['function'])
                    node1 = {'nid': 1, 'id': sub_range_r, 'node_type': 'class', 'question_node': 1,
                             'function': "none"}
                    G.add_node(node1['nid'], id=node1['id'], type=node1['node_type'],
                               question=node1['question_node'],
                               function=node1['function'])

                    G.add_edge(0, 1, relation=r, reverse=False, visited=False)
                    G.add_edge(1, 0, relation=r, reverse=True, visited=False)

                    G1 = deepcopy(G)

                    lf = none_function(G, 1)
                    lfs.append(lf)

                    lf = count_function(G1, 1)
                    lfs.append(lf)

    return lfs


# --------------------------- for grailqa enumeration --------------------

PRUNED_SUBCLASSES = defaultdict(lambda: [])
def resolve_cvt_sub_classes(domain_r, dataset):
    if domain_r in PRUNED_SUBCLASSES:
        return PRUNED_SUBCLASSES[domain_r]

    sub_domains = list(subclasses[domain_r])
    sub_domains = [d for d in sub_domains if legal_domain(d, dataset)]
    if domain_r not in sub_domains:
        sub_domains.append(domain_r)
    PRUNED_SUBCLASSES[domain_r] = sub_domains
    return sub_domains

def grail_rm_redundancy_adjancent_relations(in_relations, out_relations, use_master): 
    if use_master is None:
        return in_relations, out_relations

    refered_dict = OntologyInfo.master_property if use_master else OntologyInfo.reverse_property
    # pruned_in = [x for x in in_relations if not (x in refered_dict and refered_dict[x] in out_relations)]
    pruned_in = [x for x in in_relations if not (x in refered_dict)]
    # pruned_out = [x for x in out_relations if not (x in refered_dict and refered_dict[x] in in_relations)]
    pruned_out = [x for x in out_relations if not (x in refered_dict)]
    # print('BEFORE', len(in_relations), len(out_relations), 'AFTER', len(pruned_in), len(pruned_out))
    return pruned_in, pruned_out

def grail_rm_redundancy_two_hop_paths(paths, use_master):
    if use_master is None:
        return paths

    refered_dict = OntologyInfo.master_property if use_master else OntologyInfo.reverse_property
    valid = []
    for p0, p1 in paths:
        r0 = p0[:-2] if p0.endswith('#R') else p0
        r1 = p1[:-2] if p1.endswith('#R') else p1
        if r0 in refered_dict or r1 in refered_dict:
            continue
        valid.append((p0, p1))
    # print('BEFORE', len(paths), 'AFTER', len(valid))
    return valid

def grail_enum_one_hop_one_entity_candidates(entity: str,
                                     use_master=True):
    if (CacheBackend.cache is not None):
        in_relations_e, out_relations_e = CacheBackend.cache.query_relations(entity)
    else:  # online executing the sparql query
        in_relations_e, out_relations_e = get_adjacent_relations(entity)
    in_relations_e, out_relations_e = grail_rm_redundancy_adjancent_relations(in_relations_e, out_relations_e, use_master=use_master)

    dataset = 'grail'
    lfs = []
    # if len(entities) == 0 or len(domains) == 0:
    #     return lfs
    if len(in_relations_e) > 0:
        for r in in_relations_e:
            if not legal_relation(r, dataset):
                continue
            domain_r = relations_info[r][0]
            sub_domains = resolve_cvt_sub_classes(domain_r, dataset)
            for sub_domain_r in sub_domains:
                G = nx.MultiDiGraph()
                node = {'nid': 0, 'id': entity, 'node_type': 'entity', 'question_node': 0,
                        'function': "none"}
                G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                        function=node['function'])
                node1 = {'nid': 1, 'id': sub_domain_r, 'node_type': 'class', 'question_node': 1,
                        'function': "none"}
                G.add_node(node1['nid'], id=node1['id'], type=node1['node_type'],
                        question=node1['question_node'],
                        function=node1['function'])

                G.add_edge(1, 0, relation=r, reverse=False, visited=False)
                G.add_edge(0, 1, relation=r, reverse=True, visited=False)

                G1 = deepcopy(G)

                lf = none_function(G, 1)
                lfs.append(lf)

                lf = count_function(G1, 1)
                lfs.append(lf)

    if len(out_relations_e) > 0:
        for r in out_relations_e:
            if not legal_relation(r, dataset):
                continue

            range_r = relations_info[r][1]
            sub_ranges = resolve_cvt_sub_classes(range_r, dataset)
            for sub_range_r in sub_ranges:
                G = nx.MultiDiGraph()
                node = {'nid': 0, 'id': entity, 'node_type': 'entity', 'question_node': 0,
                        'function': "none"}
                G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                            function=node['function'])
                node1 = {'nid': 1, 'id': sub_range_r, 'node_type': 'class', 'question_node': 1,
                            'function': "none"}
                G.add_node(node1['nid'], id=node1['id'], type=node1['node_type'],
                            question=node1['question_node'],
                            function=node1['function'])

                G.add_edge(0, 1, relation=r, reverse=False, visited=False)
                G.add_edge(1, 0, relation=r, reverse=True, visited=False)

                G1 = deepcopy(G)

                lf = none_function(G, 1)
                lfs.append(lf)

                lf = count_function(G1, 1)
                lfs.append(lf)

    return lfs

# domains with subtypes > 50
GRAIL_GIANT_CVT = ['common.topic', 'location.location', 'location.administrative_division', 'people.person', 'organization.organization', 'time.event', 'freebase.unit_profile']

def _grail_valid_intermediate_type_for_joining(intermidiate_type):
    if intermidiate_type == 'type.int' or intermidiate_type == 'type.float' or intermidiate_type == 'type.datetime':
        return False
    if intermidiate_type in GRAIL_GIANT_CVT:
        return False
    # ban cvt
    # sub_domains = resolve_cvt_sub_classes(intermidiate_type, 'grail')
    # if len(sub_domains) > 1:
    #     return False
    return True

def grail_enum_two_hop_one_entity_candidates(entity: str, use_master=True):
    if (CacheBackend.cache is not None):
        paths = CacheBackend.cache.query_two_hop_paths(entity)
    else:
        paths = get_2hop_relations(entity)[2]
    paths = grail_rm_redundancy_two_hop_paths(paths, use_master)
    lfs = []
    dataset = 'grail'
    for path in paths:
        if path[0][-2:] == "#R":
            if not legal_relation(path[0][:-2], dataset):
                continue
            relation0 = '(R ' + path[0][:-2] + ')'
            intermidiate_type = relations_info[path[0][:-2]][1]
        else:
            if not legal_relation(path[0], dataset):
                continue
            relation0 = path[0]
            intermidiate_type = relations_info[path[0]][0]

        if not _grail_valid_intermediate_type_for_joining(intermidiate_type):
            continue

        if path[1][-2:] == "#R":
            if not legal_relation(path[1][:-2], dataset):
                continue
            typ = relations_info[path[1][:-2]][1]
            relation1 = '(R ' + path[1][:-2] + ')'
        else:
            if not legal_relation(path[1], dataset):
                continue
            typ = relations_info[path[1]][0]
            relation1 = path[1]
        # sub_typs = resolve_cvt_sub_classes(typ , dataset)
        # if len(sub_typs) > 10: print('WARNING LARGE sub types', typ, len(sub_typs))
        sub_typs = [typ]
        for sub_t in sub_typs:
            lf = '(AND ' + sub_t + ' (JOIN ' + relation1 + ' (JOIN ' + relation0 + ' ' + entity + ')))'
            lfs.append(lf)
            lf = '(COUNT (AND ' + sub_t + ' (JOIN ' + relation1 + ' (JOIN ' + relation0 + ' ' + entity + '))))'
            lfs.append(lf)

        # G = build_graph_from_path(entity, path)
        # G1 = deepcopy(G)
        # lf = none_function(G, 2)
        # lfs.append(lf)
        # lf = count_function(G1, 2)
        # lfs.append(lf)
    return lfs

def grail_enum_one_hop_path_with_type(entity, use_master=True):
    # dataset
    if (CacheBackend.cache is not None):
        in_relations_e, out_relations_e = CacheBackend.cache.query_relations(entity)
    else:  # online executing the sparql query
        in_relations_e, out_relations_e = get_adjacent_relations(entity)
    in_relations_e, out_relations_e = grail_rm_redundancy_adjancent_relations(in_relations_e, out_relations_e, use_master=use_master)

    dataset = 'grail'
    path_type_pairs = []
    for r in in_relations_e:
        if not legal_relation(r, dataset):
            continue
        type_r = relations_info[r][0]
        path = (r,)
        path_type_pairs.append((path, type_r))
    for r in out_relations_e:
        if not legal_relation(r, dataset):
            continue
        type_r = relations_info[r][1]
        path = (f'(R {r})',)
        path_type_pairs.append((path, type_r))

    return path_type_pairs

def grail_enum_two_hop_path_with_type(entity, use_master=True):
    if  (CacheBackend.cache is not None):
        paths = CacheBackend.cache.query_two_hop_paths(entity)
    else:
        paths = get_2hop_relations(entity)[2]
    paths = grail_rm_redundancy_two_hop_paths(paths, use_master)

    dataset = 'grail'
    path_type_pairs = []
    for path in paths:
        if path[0][-2:] == "#R":
            if not legal_relation(path[0][:-2], dataset):
                continue
            relation0 = '(R ' + path[0][:-2] + ')'
            intermidiate_type = relations_info[path[0][:-2]][1]
        else:
            if not legal_relation(path[0], dataset):
                continue
            relation0 = path[0]
            intermidiate_type = relations_info[path[0]][0]

        if not _grail_valid_intermediate_type_for_joining(intermidiate_type):
            continue

        if path[1][-2:] == "#R":
            if not legal_relation(path[1][:-2], dataset):
                continue
            typ = relations_info[path[1][:-2]][1]
            relation1 = '(R ' + path[1][:-2] + ')'
        else:
            if not legal_relation(path[1], dataset):
                continue
            typ = relations_info[path[1]][0]
            relation1 = path[1]
        path = (relation0, relation1)
        path_type_pairs.append((path, typ))
    return path_type_pairs

def is_value_domain(r):
    return r == 'type.int' or r == 'type.float' or r == 'type.datetime'

def grail_path_to_expr_fragment(entity, paths):
    if len(paths) == 1:
        return '(JOIN {} {})'.format(paths[0], entity)
    elif len(paths) == 2:
        return '(JOIN {} (JOIN {} {}))'.format(paths[1], paths[0], entity)
    else:
        raise RuntimeError('Path Lens more than 2')

def grail_merge_paths_by_ending_type(entity0, paths0, entity1, paths1):
    lfs = []
    for (p0, t0) in paths0:
        for (p1, t1) in paths1:
                        
            if not t0 == t1:
                continue
            # ending type datetime float not supported
            if is_value_domain(t0):
                continue
            typ = t0
            # dealwith overlapping prefix
            if len(p0) == 2 and len(p1) == 2 and p0[1] == p1[1]:
                # print('MERGE', len(p0), len(p1), p0, p1, typ)
                clause = '(AND (JOIN {} {}) (JOIN {} {}))'.format(p0[0], entity0, p1[0], entity1)
                expr_base = '(JOIN {} {})'.format(p0[1], clause)
            # disable two two-hop connections
            elif len(p0) == 2 and len(p1) == 2 and p0[1] != p1[1]:
                continue
            else:
                # print('END', len(p0), len(p1), p0, p1, typ)
                clause0 = grail_path_to_expr_fragment(entity0, p0)
                clause1 = grail_path_to_expr_fragment(entity1, p1)
                expr_base = '(AND {} {})'.format(clause0, clause1)
            
            lf = '(AND {} {})'.format(typ, expr_base)
            lfs.append(lf)
            lf = '(COUNT (AND {} {}))'.format(typ, expr_base)
            lfs.append(lf)
    return lfs

def grail_enum_two_entity_candidates(entity0, entity1, use_master=True):
    e0_one_hop_paths = grail_enum_one_hop_path_with_type(entity0, use_master=use_master)
    e0_two_hop_paths = grail_enum_two_hop_path_with_type(entity0, use_master=use_master)

    e1_one_hop_paths = grail_enum_one_hop_path_with_type(entity1, use_master=use_master)
    e1_two_hop_paths = grail_enum_two_hop_path_with_type(entity1, use_master=use_master)

    lfs = []
    lfs.extend(grail_merge_paths_by_ending_type(entity0, e0_one_hop_paths, entity1, e1_one_hop_paths))
    lfs.extend(grail_merge_paths_by_ending_type(entity0, e0_one_hop_paths, entity1, e1_two_hop_paths))
    lfs.extend(grail_merge_paths_by_ending_type(entity0, e0_two_hop_paths, entity1, e1_one_hop_paths))
    lfs.extend(grail_merge_paths_by_ending_type(entity0, e0_two_hop_paths, entity1, e1_two_hop_paths))
    return lfs

def _grail_canonicalize_expr(ast, use_master):
    # if ast.construction == 'R':
    #     print(ast.logical_form())
    if ast.construction == 'R' and ast.fields[0].construction == 'SCHEMA':
        if not use_master and ast.fields[0].val in OntologyInfo.reverse_property:
            new_val = OntologyInfo.reverse_property[ast.fields[0].val]
            return SchemaNode(new_val, ast.data_type, [])
        elif use_master and ast.fields[0].val in OntologyInfo.master_property:
            new_val = OntologyInfo.master_property[ast.fields[0].val]
            return SchemaNode(new_val, ast.data_type, [])
        else:
            return ast
    elif ast.construction == 'SCHEMA':
        if not use_master and ast.val in OntologyInfo.reverse_property:
            new_val = OntologyInfo.reverse_property[ast.val]
            new_node = RNode(ast.data_type, [SchemaNode(new_val, ast.data_type, [])])
            return new_node
        elif use_master and ast.val in OntologyInfo.master_property:
            new_val = OntologyInfo.master_property[ast.val]
            new_node = RNode(ast.data_type, [SchemaNode(new_val, ast.data_type, [])])
            return new_node
        else:
            return ast
    else:
        ast.fields = [_grail_canonicalize_expr(x, use_master=use_master) for x in ast.fields]
        return ast

def grail_canonicalize_expr(s_expr, use_master=True):
    if use_master is None:
        return s_expr
    ast = parse_s_expr(s_expr)
    c_ast = _grail_canonicalize_expr(ast, use_master=use_master)
    c_ast.assign_depth_and_level()
    return c_ast.compact_logical_form()

# ----------------------------------------------------------------

# --------------------------- for webqsp enumeration --------------------

def webqsp_legal_relation(r, num_entity_envolved):
    if r.endswith('#R'):
        r = r[:-2]
    # if dataset == 'webqsp':
    if r not in relations_info or r.startswith('common.') or r.startswith('type.') or r.startswith('kg.') or r.startswith('user.'):
        return False
    if num_entity_envolved == 2 and r.startswith('base.'):
        return False
    return True


def webqsp_enum_one_hop_one_entity_candidates(entity: str):

    if (CacheBackend.cache is not None):
        in_relations_e, out_relations_e = CacheBackend.cache.query_relations(entity)
    else:  # online executing the sparql query
        in_relations_e, out_relations_e = get_adjacent_relations(entity)

    lfs = []
    if len(in_relations_e) > 0:
        for r in in_relations_e:
            # print(r in relations_info, len(relations_info))
            if not webqsp_legal_relation(r, 1):
                continue
            lf = '(JOIN {} {})'.format(r, entity)
            lfs.append(lf)
    if len(out_relations_e) > 0:
        for r in out_relations_e:
            if not webqsp_legal_relation(r, 1):
                continue
            lf = '(JOIN (R {}) {})'.format(r, entity)
            lfs.append(lf)
    return lfs

def webqsp_enum_two_hop_one_entity_candidates(entity: str):
    if  (CacheBackend.cache is not None):
        paths = CacheBackend.cache.query_two_hop_paths(entity)
    else:
        paths = get_2hop_relations(entity)[2]
    lfs = []
    for path in paths:
        if path[0][-2:] == "#R":
            if not webqsp_legal_relation(path[0][:-2], 1):
                continue
            relation0 = '(R ' + path[0][:-2] + ')'
        else:
            if not webqsp_legal_relation(path[0], 1):
                continue
            relation0 = path[0]
        if path[1][-2:] == "#R":
            if not webqsp_legal_relation(path[1][:-2], 1):
                continue
            relation1 = '(R ' + path[1][:-2] + ')'
        else:
            if not webqsp_legal_relation(path[1], 1):
                continue
            relation1 = path[1]
        lf = '(JOIN ' + relation1 + ' (JOIN ' + relation0 + ' ' + entity + '))'
        lfs.append(lf)
    return lfs

def webqsp_enum_two_entity_candidates(e0, e1):
    if (CacheBackend.cache is not None):
        paths0 = CacheBackend.cache.query_two_hop_paths(e0)
        paths1 = CacheBackend.cache.query_two_hop_paths(e1)
    else:
        paths0 = get_2hop_relations(e0)[2]
        paths1 = get_2hop_relations(e1)[2]

    lfs = []
    for p0 in paths0:
        for p1 in paths1:
            if not (p0[1] == p1[1]):
                continue
            if not webqsp_legal_relation(p0[1], 2) or not webqsp_legal_relation(p0[0], 2) or not webqsp_legal_relation(p1[0], 2):
                continue
            # AND r e, r template
            lowa, lowb = p0[0], p1[0]
            ea, eb = e0, e1
            if lowa.endswith('#R') and not lowb.endswith('#R'):
                lowa, lowb = lowb, lowa
                ea, eb = eb, ea

            if lowa.endswith('#R'):
                lowa = '(R ' + lowa[:-2] + ')'
            if lowb.endswith('#R'):
                lowb = '(R ' + lowb[:-2] + ')'
            # ( AND ( JOIN SCHEMA ENTITY ) ( JOIN ( R SCHEMA ) ENTITY ) )
            base_lf = '(AND (JOIN {} {}) (JOIN {} {}))'.format(lowa, ea, lowb, eb)
            lfs.append(base_lf)
            # JOIN R (AND ( ))
            rhigh = p0[1]
            if rhigh.endswith('#R'):
                rhigh = '(R ' + rhigh[:-2] + ')'
            joint_lf = '(JOIN {} {})'.format(rhigh, base_lf)
            lfs.append(joint_lf)
    return lfs
# -----------------------------------------------

def build_grail_cache_from_lagacy_cache():

    # relations
    with open('cache/1hop_in_r', 'r') as f:
        in_relations = json.load(f)

    with open('cache/1hop_out_r', 'r') as f:
        out_relations = json.load(f)

    relation_data = {}
    for k in in_relations:
        if k not in out_relations:
            continue
        in_r = in_relations[k]
        out_r = out_relations[k]
        relation_data[k] = (in_r, out_r)
    print(len(relation_data))
    dump_json(relation_data,  join('cache', 'grail-LinkedRelation.bin'))

    # paths
    with open('cache/2hop_paths', 'r') as f:
        two_hop_paths = json.load(f)
    path_data = {}
    for k, v in two_hop_paths.items():
        tup_v = [tuple(x) for x in v]
        path_data[k] = tup_v
    print(len(path_data))
    dump_json(path_data,  join('cache', 'grail-TwoHopPath.bin'))

