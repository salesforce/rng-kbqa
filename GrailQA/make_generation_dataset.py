"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import json
import torch
from components.utils import *
from components.expr_parser import tokenize_s_expr
import editdistance
import argparse

FORCE_DIFF_CONFIG=False

def load_candidates_file(fname):
    info = []
    with open(fname) as f:
        for l in f:
            info.append(json.loads(l))
    return info

def main(split, logit_file, is_training=False, num_retain=10, force_diff=FORCE_DIFF_CONFIG):
    candidate_info = load_candidates_file(f'outputs/grail_{split}_candidates-ranking.jsonline')
    logit_info = torch.load(logit_file)

    gen_dataset = []
    num_spec = 0
    top_ex_cnt = 0
    for data in candidate_info:
        target_expr = data['target_expr']
        candidates = data['candidates']
        if not candidates:
            continue
        qid = data['qid']
        if qid not in logit_info:
            continue
        logits = logit_info[qid]
        # top_idx = sorted(list(range(len(logits))), key=lambda x: logits[x], reverse=True)[:num_retain]
        
        sorted_idx = torch.argsort(-logits).tolist()
        if not FORCE_DIFF_CONFIG:
            top_idx = sorted_idx[:num_retain]
        else:
            raise RuntimeError('Not supported yet')

        top_candidates = [candidates[i] for i in top_idx]
        top_pred, top_is_ex = top_candidates[0]['logical_form'], top_candidates[0]['ex']

        # if find a good alternative, continue
        # target_approx == target_gt means no approximation 
        if top_pred != target_expr and top_is_ex:
            gen_target = top_pred
            # special case
            num_spec += 1
        else:
            gen_target = target_expr
        
        if top_is_ex:
            top_ex_cnt += 1
        gen_ex = {'qid': qid, 'genation_target': gen_target, 'top_candidates': top_candidates, 'target_full_expr': target_expr}
        # print(qid)
        # print('Gen Targ', gen_target)
        # print('Top Pred', top_pred)
        # print(top_candidates)

        gen_dataset.append(gen_ex)

    print(len(gen_dataset))
    print(num_spec/len(gen_dataset))
    print(top_ex_cnt/len(gen_dataset))
    dump_json(gen_dataset, f'outputs/grail_{split}_gen.json')

def main_random(split, logit_file, is_training=False, num_retain=10, force_diff=FORCE_DIFF_CONFIG):
    import random
    random.seed(123)
    candidate_info = load_candidates_file(f'outputs/grail_{split}_candidates-ranking.jsonline')
    logit_info = torch.load(logit_file)

    gen_dataset = []
    num_spec = 0
    top_ex_cnt = 0
    for data in candidate_info:
        target_expr = data['target_expr']
        candidates = data['candidates']
        if not candidates:
            continue
        qid = data['qid']
        if qid not in logit_info:
            continue
        logits = logit_info[qid]
        # top_idx = sorted(list(range(len(logits))), key=lambda x: logits[x], reverse=True)[:num_retain]
        
        sorted_idx = torch.argsort(-logits).tolist()
        # if not FORCE_DIFF_CONFIG:
        #     top_idx = sorted_idx[:num_retain]
        # else:
        #     raise RuntimeError('Not supported yet')
        top_idx = random.sample(range(len(sorted_idx)), min(num_retain,len(sorted_idx)) )

        top_candidates = [candidates[i] for i in top_idx]
        top_pred, top_is_ex = top_candidates[0]['logical_form'], top_candidates[0]['ex']

        # if find a good alternative, continue
        # target_approx == target_gt means no approximation 
        if top_pred != target_expr and top_is_ex:
            gen_target = top_pred
            # special case
            num_spec += 1
        else:
            gen_target = target_expr
        
        if top_is_ex:
            top_ex_cnt += 1
        gen_ex = {'qid': qid, 'genation_target': gen_target, 'top_candidates': top_candidates, 'target_full_expr': target_expr}
        # print(qid)
        # print('Gen Targ', gen_target)
        # print('Top Pred', top_pred)
        # print(top_candidates)

        gen_dataset.append(gen_ex)

    print(len(gen_dataset))
    print(num_spec/len(gen_dataset))
    print(top_ex_cnt/len(gen_dataset))
    dump_json(gen_dataset, f'outputs/grail_{split}_randgen.json')


def inspect_generation_dataset(split, cutoff=5):
    gen_data =  load_json(f'outputs/grail_{split}_gen.json')
    covered_cnt = 0
    top_ex_cnt = 0
    num_close = 0
    for data in gen_data:
        gt = data['genation_target']
        candidates = data['top_candidates'][:cutoff]
        if any([x['ex'] for x in candidates]):
            covered_cnt += 1
        if candidates[0]['ex']:
            top_ex_cnt += 1
        top_pred = candidates[0]['logical_form']
        if any([x['ex'] for x in candidates]) and not candidates[0]['ex']:
            print('---------COVER NOT EX--------')
            print(gt)
            for x in candidates:
                print('\t', x)
        gt_tokens = tokenize_s_expr(gt)
        top_tokens = tokenize_s_expr(top_pred)
        edit_dist = editdistance.eval(gt_tokens, top_tokens)
        if edit_dist == 1:
            print('--------EDIST 1---------')
            print(gt)
            print(top_pred)
            num_close += 1
        
    print(covered_cnt / len(gen_data))
    print(top_ex_cnt / len(gen_data))
    print(num_close / len(gen_data))

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--logit_file', default=None, help='logit file')
    args = parser.parse_args()
    if args.logit_file is None:
        args.logit_file = f'misc/grail_{args.split}_candidate_logits.bin'

    print('split', args.split, 'logit_file', args.logit_file)
    return args

if __name__=='__main__':
    args = _parse_args()
    main(args.split, args.logit_file)
