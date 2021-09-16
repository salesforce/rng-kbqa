#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

PWD=$(pwd)
echo ${PWD}

setup_dataset () {
    DATASET=$1
    echo "SETTING UP"${DATASET}
    ln -s ${PWD}/framework/components ${PWD}/${DATASET}
    ln -s ${PWD}/framework/models ${PWD}/${DATASET}
    ln -s ${PWD}/framework/executor ${PWD}/${DATASET}
    ln -s ${PWD}/framework/ontology ${PWD}/${DATASET}

    ln -s ${PWD}/framework/run_ranker.py ${PWD}/${DATASET}/
    ln -s ${PWD}/framework/run_generator.py ${PWD}/${DATASET}/
}

setup_dataset GrailQA
setup_dataset WebQSP
