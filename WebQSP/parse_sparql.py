"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import pickle
import json
import os
import random
from re import L
import shutil
from collections import Counter
from components.utils import *
from components.expr_parser import parse_s_expr, extract_entities, tokenize_s_expr
from executor.sparql_executor import execute_query
from executor.sparql_executor import get_label, execute_query
from executor.logic_form_util import lisp_to_sparql

class ParseError(Exception):
    pass

class Parser:
    def __init__(self):
        pass

    def parse_query(self, query, topic_mid):
        # print('QUERY', query)
        lines = query.split('\n')
        lines = [x for x in lines if x]

        assert lines[0] != '#MANUAL SPARQL'

        prefix_stmts = []        
        line_num = 0
        while True:
            l = lines[line_num]
            if l.startswith('PREFIX'):
                prefix_stmts.append(l)
            else:
                break
            line_num = line_num + 1

        next_line = lines[line_num]
        assert next_line.startswith('SELECT DISTINCT ?x')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'WHERE {'
        assert lines[-1] in ['}', 'LIMIT 1']

        lines = lines[line_num :]
        assert all(['FILTER (str' not in x for x in lines])
        # normalize body lines

        body_lines, spec_condition = self.normalize_body_lines(lines)
        # assert all([x.startswith('?') or x.startswith('ns') or x.startswith('FILTER') for x in body_lines])
        # we only parse query following this format
        if body_lines[0].startswith('FILTER'):
            predefined_filter0 = body_lines[0]
            predefined_filter1 = body_lines[1]
            assert predefined_filter0 == f'FILTER (?x != ns:{topic_mid})'
            assert predefined_filter1 == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
            # if predefined_filter0 != f'FILTER (?x != ns:{topic_mid})':
            #     print('QUERY', query)
            #     print('First Filter')
            # if predefined_filter1 != "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
            #     print('QUERY', query)
            #     print('Second Filter')
            # if any([not (x.startswith('?') or x.startswith('ns:')) for x in body_lines]):
            #     print('Unprincipled Filter')
            #     print('QUERY', query)
            body_lines = body_lines[2:]

        assert all([(x.startswith('?') or x.startswith('ns:')) for x in body_lines])
        # print(body_lines)
        var_dep_list = self.parse_naive_body(body_lines, '?x')
        s_expr = self.dep_graph_to_s_expr(var_dep_list, '?x', spec_condition)
        return s_expr

    def normalize_body_lines(self, lines):
        if lines[-1] == 'LIMIT 1':
            # spec_condition = argmax
            # who did jackie robinson first play for?
            # WHERE {
            # ns:m.0443c ns:sports.pro_athlete.teams ?y .
            # ?y ns:sports.sports_team_roster.team ?x .
            # ?y ns:sports.sports_team_roster.from ?sk0 .
            # }
            # ORDER BY DESC(xsd:datetime(?sk0))
            # LIMIT 1
            order_line = lines[-2]
            direction = 'argmax' if 'DESC(' in order_line else 'argmin'
            # assert '?sk0' in
            # print(line)
            assert ('?sk0' in order_line)
            _tmp_body_lines = lines[1:-3]
            body_lines = []
            hit = False
            for l in _tmp_body_lines:
                if '?sk0' in l:
                    self.parse_assert(l.endswith('?sk0 .') and not hit)
                    hit = True
                    arg_var, arg_r = l.split(' ')[0], l.split(' ')[1]
                    arg_r = arg_r[3:] #rm ns:
                else:
                    body_lines.append(l)
                    
            return body_lines, (direction, arg_var, arg_r)
        # check if xxx
        elif lines[-4].startswith('FILTER(NOT EXISTS {?'):
            # WHERE {
            # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
            # ?y ns:government.government_position_held.office_holder ?x .
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} || 
            # EXISTS {?y ns:government.government_position_held.from ?sk1 . 
            # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} || 
            # EXISTS {?y ns:government.government_position_held.to ?sk3 . 
            # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
            # }
            body_lines = lines[1:-7]
            range_lines = lines[-7:-1]
            range_prompt = range_lines[0]
            range_prompt = range_prompt[range_prompt.index('{') + 1:range_prompt.index('}')]
            range_var = range_prompt.split(' ')[0]
            range_relation = range_prompt.split(' ')[1]
            range_relation = '.'.join(range_relation.split('.')[:2]) + '.time_macro'
            range_relation = range_relation[3:] # rm ns:

            range_start = range_lines[2].split(' ')[2]
            range_start = range_start[1:]
            range_start = range_start[:range_start.index('"')]
            range_end = range_lines[5].split(' ')[2]
            range_end = range_end[1:]
            range_end = range_end[:range_end.index('"')]
            assert range_start[:4] == range_end[:4]
            range_year = range_start[:4] + '^^http://www.w3.org/2001/XMLSchema#date' #to fit parsable
            return body_lines, ('range', range_var, range_relation, range_year)
        else:
            body_lines = lines[1:-1]
            return body_lines, None        
        

    def dep_graph_to_s_expr(self, var_dep_list, ret_var, spec_condition=None):
        self.parse_assert(var_dep_list[0][0] == ret_var)
        var_dep_list.reverse()
        parsed_dict = {}

        spec_var = spec_condition[1] if spec_condition is not None else None

        for var_name, dep_relations in var_dep_list:
            # expr = ''
            dep_relations[0]
            clause = self.triplet_to_clause(var_name,  dep_relations[0], parsed_dict)
            for tri in dep_relations[1:]:
                n_clause = self.triplet_to_clause(var_name, tri, parsed_dict)
                clause = 'AND ({}) ({})'.format(n_clause, clause)
            if var_name == spec_var:
                if  spec_condition[0] == 'argmax' or spec_condition[0] == 'argmin':
                    relation = spec_condition[2]
                    clause = '{} ({}) {}'.format(spec_condition[0].upper(), clause, relation)
                elif spec_condition[0] == 'range':
                    relation, time_point = spec_condition[2], spec_condition[3]
                    n_clause = 'JOIN {} {}'.format(relation, time_point)
                    clause = 'AND ({}) ({})'.format(n_clause, clause)
            parsed_dict[var_name] = clause
        return '(' + parsed_dict[ret_var] + ')'

    def triplet_to_clause(self, tgt_var, triplet, parsed_dict):
        if triplet[0] == tgt_var:
            this = triplet[0]
            other = triplet[-1]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN {} {}'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            this = triplet[-1]
            other = triplet[0]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN (R {}) {}'.format(triplet[1], other)
        else:
            raise ParseError()


    def parse_assert(self, eval):
        if not eval:
            raise ParseError()

    def parse_naive_body(self, body_lines, ret_var):
        # ret_variable
        # body_lines
        assert all([x[-1] == '.' for x in body_lines])
        triplets = [x.split(' ') for x in body_lines]
        triplets = [x[:-1] for x in triplets]

        # remove ns 
        triplets = [[x[3:] if x.startswith('ns:') else x for x in tri] for tri in triplets]
        # dependancy graph
        triplets_pool = triplets
        # while True:
        var_dep_list = []
        successors = []
        dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))        
        # vars_pool = []
        # go over un resolved vars
        # for tri in triplets_pool:
        #     if tri[0].startswith('?') and tri[0] not in vars_pool and tri[0] != ret_var:
        #         vars_pool.append(tri[0])
        #     if tri[-1].startswith('?') and tri[-1] not in vars_pool and tri[-1] != ret_var:
        #         vars_pool.append(tri[-1])
        
        # for tgt_var in vars_pool:
        #     dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, tgt_var)
        #     self.parse_assert(len(dep_triplets) > 0)
        #     var_dep_list.append((tgt_var, dep_triplets))
        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, tgt_var, successors)
            self.parse_assert(len(dep_triplets) > 0)
            var_dep_list.append((tgt_var, dep_triplets))

        self.parse_assert(len(triplets_pool) == 0)
        return var_dep_list
        

    def resolve_dependancy(self, triplets, target_var, successors):
        dep = []
        left = []
        for tri in triplets:
            if tri[0] == target_var:
                dep.append(tri)
                if tri[-1].startswith('?') and tri[-1] not in successors:
                    successors.append(tri[-1])
            elif tri[-1] == target_var:
                dep.append(tri)
                if tri[0].startswith('?') and tri[0] not in successors:
                    successors.append(tri[0])
            else:
                left.append(tri)
        return dep, left

class SparqlParse:
    def __init__(self):
        select_stmt = None
        prefix_stmts = None
        where_stmts = None
        query_stmts = None

def convert_parse_instance(parse):
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        s_expr = parser.parse_query(sparql, parse['TopicEntityMid'])
        # print('---GOOD------')
        # print(sparql)
        # print(s_expr)
    except AssertionError:
        s_expr = 'null'
    # print(parse[''])
    parse['SExpr'] = s_expr
    return parse, s_expr != 'null'

# def s_expr_to_sparql(sparql_query):
#     print(sparql_query)

def webq_s_expr_to_sparql_query(s_expr):
    ast = parse_s_expr(s_expr)

def execute_webq_s_expr(s_expr):
    try:
        sparql_query = lisp_to_sparql(s_expr)
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    return denotation

def augment_with_s_expr(split):
    # dataset = load_json('WebQSP.train.json')
    dataset = load_json(f'outputs/WebQSP.{split}.json')
    dataset = dataset['Questions']
    total_num = 0
    hit_num = 0
    for data in dataset:
        aug_parses = []
        for parse in data['Parses']:
            total_num += 1
            instance, flag_success = convert_parse_instance(parse)
            aug_parses.append(instance)
            if flag_success:
                hit_num += 1
        data['Parses'] = aug_parses
    print(hit_num, total_num, hit_num/total_num, len(dataset))
    dump_json(dataset, f'outputs/WebQSP.{split}.expr.json', indent=2)

def execution_sanity_check(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    # dataset = dataset['Questions']
    dismatch_cnt = 0
    for i, data in enumerate(dataset):
        for parse in data['Parses']:
            # results = execute_query(parse['Sparql'])
            sparql = parse['Sparql']
            s_expr = parse['SExpr']
            if s_expr == 'null':
                continue
            # print(s_expr)
            # print(sparql)
            results = execute_webq_s_expr(s_expr)
            gt = [a['AnswerArgument'] for a in parse['Answers']]
            if set(results) != set(gt):
                dismatch_cnt += 1
                print('Dismatch at', i, dismatch_cnt)
                print(sparql)
                print(s_expr)
                print(results)
                print(gt)
    print(dismatch_cnt)

def entity_count_stats_check(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    # dataset = dataset['Questions']
    entity_counts = []
    for i, data in enumerate(dataset):
        for parse in data['Parses']:
            # results = execute_query(parse['Sparql'])
            sparql = parse['Sparql']
            s_expr = parse['SExpr']
            if s_expr == 'null':
                continue
            # print(s_expr)
            # print(sparql)
            entity_counts.append(len(extract_entities(s_expr)))
    count_dict = Counter(entity_counts)
    print(count_dict)
    print(len(count_dict))
    for k, v in count_dict.items():
        print(k, v/len(entity_counts))


def extract_used_relation(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    # dataset = dataset['Questions']
    used_relations = set()
    for i, data in enumerate(dataset):
        for parse in data['Parses']:
            # results = execute_query(parse['Sparql'])
            sparql = parse['Sparql']
            s_expr = parse['SExpr']
            if s_expr == 'null':
                continue
            toks = tokenize_s_expr(s_expr)
            relations = [x for x in toks if '.' in x and not x.startswith('m.') and '^^http' not in x]
            [used_relations.add(r) for r in relations]
            # exit()
    print(split, len(used_relations))
    # dump_to_bin(used_relations, f'misc/{split}_relations.bin')
    return used_relations

def find_macro_template_from_query(query, topic_mid):
    # print('QUERY', query)
    lines = query.split('\n')
    lines = [x for x in lines if x]

    assert lines[0] != '#MANUAL SPARQL'

    prefix_stmts = []        
    line_num = 0
    while True:
        l = lines[line_num]
        if l.startswith('PREFIX'):
            prefix_stmts.append(l)
        else:
            break
        line_num = line_num + 1

    next_line = lines[line_num]
    assert next_line.startswith('SELECT DISTINCT ?x')
    line_num = line_num + 1
    next_line = lines[line_num]
    assert next_line == 'WHERE {'
    assert lines[-1] in ['}', 'LIMIT 1']

    lines = lines[line_num :]
    assert all(['FILTER (str' not in x for x in lines])
    # normalize body lines
    # return_val = check_time_macro_from_body_lines(lines)
    # if return_val:
        
    # relation_prefix, suffix_pair = c
    return check_time_macro_from_body_lines(lines)

def check_time_macro_from_body_lines(lines):
    # check if xxx
    if lines[-4].startswith('FILTER(NOT EXISTS {?'):
        # WHERE {
        # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
        # ?y ns:government.government_position_held.office_holder ?x .
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} || 
        # EXISTS {?y ns:government.government_position_held.from ?sk1 . 
        # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} || 
        # EXISTS {?y ns:government.government_position_held.to ?sk3 . 
        # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
        # }
        body_lines = lines[1:-7]
        range_lines = lines[-7:-1]
        range_prompt_start = range_lines[0]
        range_prompt_start = range_prompt_start[range_prompt_start.index('{') + 1:range_prompt_start.index('}')]
        range_relation_start = range_prompt_start.split(' ')[1]
        
        # range_relation = '.'.join(range_relation.split('.')[:2]) + '.time_macro'
        # range_relation = range_relation[3:] # rm ns:

        range_prompt_end = range_lines[3]
        range_prompt_end = range_prompt_end[range_prompt_end.index('{') + 1:range_prompt_end.index('}')]
        range_relation_end = range_prompt_end.split(' ')[1]
        
        assert range_relation_start.split('.')[:2] == range_relation_end.split('.')[:2]
        start_suffix = range_relation_start.split('.')[-1]
        end_suffix = range_relation_end.split('.')[-1]
        prefix = '.'.join(range_relation_start.split('.')[:2])[3:]
        return prefix, start_suffix, end_suffix
    else:
        return None
    

def extract_macro_template_from_instance(parse):
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        return find_macro_template_from_query(sparql, parse['TopicEntityMid'])
    except AssertionError:
        return None

def extract_macro_template(split):
    # dataset = load_json('WebQSP.train.json')
    dataset = load_json(f'outputs/WebQSP.{split}.json')
    dataset = dataset['Questions']
    
    templates = set()
    for data in dataset:
        aug_parses = []
        for parse in data['Parses']:
            return_val = extract_macro_template_from_instance(parse)
            if return_val is not None:
                templates.add(return_val)
    
    print(len(templates))
    templates = sorted(list(templates))
    print(templates)


train_templates = [('american_football.football_historical_coach_position', 'from', 'to'), ('architecture.ownership', 'start_date', 'end_date'),
 ('award.award_honor', 'year', 'year'), ('business.employment_tenure', 'from', 'to'), ('business.sponsorship', 'from', 'to'),
 ('celebrities.romantic_relationship', 'start_date', 'end_date'), ('chemistry.chemical_element', 'discovery_date', 'discovery_date'),
 ('film.film', 'initial_release_date', 'initial_release_date'),('government.government_position_held', 'from', 'to'),
 ('law.invention', 'date_of_invention', 'date_of_invention'), ('law.judicial_tenure', 'from_date', 'to_date'),
 ('organization.organization_relationship', 'to', 'from'), ('people.marriage', 'from', 'to'),
 ('people.place_lived', 'end_date', 'start_date'), ('sports.sports_team_coach_tenure', 'from', 'to'),
 ('sports.sports_team_roster', 'from', 'to'), ('sports.team_venue_relationship', 'from', 'to'),
 ('time.event', 'start_date', 'end_date'), ('tv.regular_tv_appearance', 'from', 'to'), ('tv.tv_network_duration', 'from', 'to')]

train_constraints = [('m.0kpys4', 'US State'), ('m.044801x', 'Professional Sports Team'), ('m.01xljyt', 'American Football team'),
('m.01m9', 'City/Town/Village'), ('m.01xpjyz', 'Airport'), ('m.025dnr9', 'American Football Conference'), ('m.01xs05k', 'River'),
('m.01xryvm', 'Book'), ('m.01mh', 'Continent'), ('m.01y2hnl', 'College/University'),('m.01xljv1', 'Super bowl'), ('m.01xxv5b', 'Island Group'),
('m.02_3pws', 'Mexican state'), ('m.025dnqw', 'American Football Division'), ('m.01y2hn6', 'School'), ('m.01n7', 'Location'),
('m.03jz7ls', 'Written Work'), ('m.08scbsj', 'Subatomic particle'), ('m.03w5clp', 'Production company'), ('m.0kpym_', 'US County'),
('m.01xljtp', 'Hospital'), ('m.04fnrhx', 'Monarch'), ('m.01xs039', 'Mountain range'), ('m.01mp', 'Country'), ('m.02knxyp', 'Religious Text'),
('m.0256985', 'Baseball Team'), ('m.05czz29', 'Brand'), ('m.01nt', 'Region'), ('m.02ht342', 'Automobile Make'), ('m.02_3phk', 'Dutch province')]

def mine_common_type_constraint(split):
    dataset = load_json(f'outputs/WebQSP.{split}.expr.json')
    # dataset = dataset['Questions']
    constrained_types = set()
    for i, data in enumerate(dataset):
        for parse in data['Parses']:
            # results = execute_query(parse['Sparql'])
            sparql = parse['Sparql']
            s_expr = parse['SExpr']
            if s_expr == 'null':
                continue
            if not s_expr.startswith('(AND (JOIN common.topic.notable_types'):
                continue
            
            toks = tokenize_s_expr(s_expr)
            cons_type = toks[5]
            constrained_types.add(cons_type)
            
    print(split, len(constrained_types))
    # dump_to_bin(used_relations, f'misc/{split}_relations.bin')
    print(constrained_types)
    return constrained_types

def make_partial_train_dev():
    random.seed(17)
    data = load_json('outputs/WebQSP.train.expr.json')
    random.shuffle(data)
    ptrain = data[:-200]
    pdev = data[-200:]
    print(len(ptrain))
    print(len(pdev))
    dump_json(ptrain, f'outputs/WebQSP.ptrain.expr.json', indent=2)
    dump_json(pdev, f'outputs/WebQSP.pdev.expr.json', indent=2)

def make_orig_partial_train_dev():
    random.seed(17)
    dataset = load_json('outputs/WebQSP.train.json')
    data = dataset['Questions']
    random.shuffle(data)
    ptrain = data[:-200]
    pdev = data[-200:]
    print(len(ptrain))
    print(len(pdev))
    dump_json({'Questions': ptrain} , f'outputs/WebQSP.ptrain.json', indent=2)
    dump_json({'Questions': pdev}, f'outputs/WebQSP.pdev.json', indent=2)


if __name__ == '__main__':
    parser = Parser()
    augment_with_s_expr('train')
    augment_with_s_expr('test')
    make_orig_partial_train_dev()
    make_partial_train_dev()
    # extract_macro_template('train')
    # mine_common_type_constraint('train'):
