"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    return toks

def extract_mentioned_entities(expr):
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
