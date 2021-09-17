import json
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from executor.logic_form_util import none_function, count_function
from executor.sparql_executor import get_adjacent_relations, get_2hop_relations

path = './'

domain_info = defaultdict(lambda: 'base')
with open(path + 'ontology/domain_info', 'r') as f:
    # domain_info = json.load(f)
    domain_info.update(json.load(f))

with open(path + 'ontology/fb_roles', 'r') as f:
    contents = f.readlines()

with open(path + 'ontology/fb_types', 'r') as f:
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


gq1 = False

if gq1:
    with open(path + 'cache/1hop_in_r_gq1', 'r') as f:
        in_relations = json.load(f)

    with open(path + 'cache/1hop_out_r_gq1', 'r') as f:
        out_relations = json.load(f)

    with open(path + 'cache/2hop_paths_gq1', 'r') as f:
        two_hop_paths = json.load(f)
else:
    with open(path + 'cache/1hop_in_r', 'r') as f:
        in_relations = json.load(f)

    with open(path + 'cache/1hop_out_r', 'r') as f:
        out_relations = json.load(f)

    with open(path + 'cache/2hop_paths', 'r') as f:
        two_hop_paths = json.load(f)


def add_node_to_G(G, node):
    G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
               function=node['function'])


def legal_relation(r):
    # print("gq1:", gq1)
    if r not in relations_info or r[:7] == 'common.' or r[:5] == 'type.' or r[:3] == 'kg.' or r[:5] == 'user.':
        return False
    if not gq1:
        if r[:5] == 'base.':
            return False
    return True

def generate_all_logical_forms_2(entity: str, offline=True):
    if offline:
        if entity in two_hop_paths:
            paths = two_hop_paths[entity]
        else:
            print('Trying online...')
            paths = get_2hop_relations(entity)[2]
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


def generate_all_logcial_forms_2_with_domain(entity: str, domain: str, offline=True):
    if offline:
        paths = two_hop_paths[entity]
    else:
        paths = get_2hop_relations(entity)[2]
    lfs = []
    for path in paths:
        if path[0].replace('#R', '') in domain_dict_relations[domain] \
                and path[1].replace('#R', '') in domain_dict_relations[domain]:
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
                                     offline=True):
    def r_in_domains(domains0, r0):
        for domain in domains0:
            if r0 in domain_dict_relations[domain]:
                return True

        return False

    if offline:
        if entity in in_relations:
            in_relations_e = in_relations[entity]
        else:
            in_relations_e = []

        if entity in out_relations:
            out_relations_e = out_relations[entity]
        else:
            out_relations_e = []

        if len(in_relations_e) + len(out_relations_e) == 0:  # then try to find the relations online
            in_relations_e, out_relations_e = get_adjacent_relations(entity)

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


def get_vocab_info_online(entity: str, step=2):  # get the 1 hop vocabulary of a given entity via Sparql
    vocab = set()
    if step == 1:
        in_relations, out_relations = get_adjacent_relations(entity)

    elif step == 2:
        in_relations, out_relations, _ = get_2hop_relations(entity)

    else:
        return None

    vocab.update(in_relations)
    vocab.update(out_relations)

    for r in in_relations:
        if r in relations_info:
            vocab.add(relations_info[r][0])
    for r in out_relations:
        if r in relations_info:
            vocab.add(relations_info[r][1])

    return vocab
