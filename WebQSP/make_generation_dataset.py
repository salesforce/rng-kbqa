"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import torch
import argparse
from components.utils import *

def make_gen_data(split, logit_file, is_training=False, num_retain=10,):
    candidate_info = load_json(f'outputs/webqsp_{split}_candidates-ranking.json')
    logit_info = torch.load(logit_file)

    gen_dataset = []
    num_spec = 0
    for data in candidate_info:
        target_approx = data['target_s_expr']
        approx_exprs = data['approx_s_expressions']
        gt_exprs = data['gt_s_expressions']
        if target_approx == 'null':
            target_gt = 'null'
            # skip when training
            if is_training:
                continue
        else:
            simplest_idx = approx_exprs.index(target_approx)
            target_gt = gt_exprs[simplest_idx]

        candidates = data['candidates']
        if not candidates:
            continue
        qid = data['qid']
        if qid not in logit_info:
            continue
        logits = logit_info[qid]
        sorted_idx = torch.argsort(-logits).tolist()
        top_idx = sorted_idx[:num_retain]
        # top_idx = sorted(list(range(len(logits))), key=lambda x: logits[x], reverse=True)[:num_retain]
        top_candidates = [candidates[i] for i in top_idx]

        top_pred, top_is_ex = top_candidates[0]['logical_form'], top_candidates[0]['ex']
        

        # if find a good alternative, continue
        # target_approx == target_gt means no approximation 
        if target_approx == target_gt and top_pred != target_approx and top_is_ex:
            gen_target = top_pred
            # special case
            # print('Special Case')
            # print('TOP', top_pred)
            # print('Target', target_gt)
            # print('ALL GT', gt_exprs)
            # print('ALL Aprox', approx_exprs)
            num_spec += 1
        else:
            gen_target = target_gt

        gen_ex = {'qid': qid, 'genation_target': gen_target, 'top_candidates': top_candidates, 'target_approx_expr': target_approx, 'target_full_expr': target_gt}
        # print(qid)
        # print('Gen Targ', gen_target)
        # print('Top Pred', top_pred)
        # print(top_candidates)

        gen_dataset.append(gen_ex)

    print(len(gen_dataset))
    print(num_spec)
    dump_json(gen_dataset, f'outputs/webqsp_{split}_gen.json')


def make_randgen_data(split, logit_file, is_training=False, num_retain=10,):
    import random
    random.seed(123)
    candidate_info = load_json(f'outputs/webqsp_{split}_candidates-ranking.json')
    logit_info = torch.load(logit_file)

    gen_dataset = []
    num_spec = 0
    for data in candidate_info:
        target_approx = data['target_s_expr']
        approx_exprs = data['approx_s_expressions']
        gt_exprs = data['gt_s_expressions']
        if target_approx == 'null':
            target_gt = 'null'
            # skip when training
            if is_training:
                continue
        else:
            simplest_idx = approx_exprs.index(target_approx)
            target_gt = gt_exprs[simplest_idx]

        candidates = data['candidates']
        if not candidates:
            continue
        qid = data['qid']
        if qid not in logit_info:
            continue
        logits = logit_info[qid]
        sorted_idx = torch.argsort(-logits).tolist()
        # top_idx = sorted_idx[:num_retain]
        top_idx = random.sample(range(len(sorted_idx)), min(num_retain,len(sorted_idx)) )
        # top_idx = sorted(list(range(len(logits))), key=lambda x: logits[x], reverse=True)[:num_retain]
        top_candidates = [candidates[i] for i in top_idx]

        top_pred, top_is_ex = top_candidates[0]['logical_form'], top_candidates[0]['ex']
        

        # if find a good alternative, continue
        # target_approx == target_gt means no approximation 
        if target_approx == target_gt and top_pred != target_approx and top_is_ex:
            gen_target = top_pred
            # special case
            # print('Special Case')
            # print('TOP', top_pred)
            # print('Target', target_gt)
            # print('ALL GT', gt_exprs)
            # print('ALL Aprox', approx_exprs)
            num_spec += 1
        else:
            gen_target = target_gt

        gen_ex = {'qid': qid, 'genation_target': gen_target, 'top_candidates': top_candidates, 'target_approx_expr': target_approx, 'target_full_expr': target_gt}
        # print(qid)
        # print('Gen Targ', gen_target)
        # print('Top Pred', top_pred)
        # print(top_candidates)

        gen_dataset.append(gen_ex)

    print(len(gen_dataset))
    print(num_spec)
    dump_json(gen_dataset, f'outputs/webqsp_{split}_randgen.json')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--logit_file', default=None, help='logit file')
    args = parser.parse_args()
    if args.logit_file is None:
        args.logit_file = f'misc/webqsp_{args.split}_candidate_logits.bin'

    print('split', args.split, 'logit_file', args.logit_file)
    return args

if __name__=='__main__':
    args = _parse_args()
    make_gen_data(args.split, args.logit_file, is_training='train' in args.split)
