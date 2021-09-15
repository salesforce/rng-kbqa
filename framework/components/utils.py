"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import pickle
import json
import os
import shutil

def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)

def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)

def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
