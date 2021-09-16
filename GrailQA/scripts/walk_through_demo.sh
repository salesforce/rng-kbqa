#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

echo "------(i) ENTITY DETECTION---------"
python detect_entity_mention.py --split demo

echo "\n\n"
echo "------(ii) ENTITY DISAMBIGUATING---------"
sh scripts/run_disamb.sh predict checkpoints/grail_bert_entity_disamb demo

echo "\n\n"
echo "------(iii) ENUM CANDIDATE---------"
python enumerate_candidates.py --split demo --pred_file misc/grail_demo_entity_linking.json

echo "\n\n"
echo "------(iv) RUN RANKER---------"
sh scripts/run_ranker.sh predict checkpoints/grail_bert_ranking demo

echo "\n\n"
echo "------(v) RUN GENERATOR---------"
python make_generation_dataset.py --split demo --logit_file misc/grail_demo_candidate_logits.bin
sh scripts/run_gen.sh predict checkpoints/grail_t5_generation demo

echo "\n\n"
echo "------(vi) FINAL INFERENCE---------"
python eval_topk_prediction.py --split demo --pred_file misc/grail_demo_topk_generations.json


echo "\n\n"
echo "------EVAL ON DEMO---------"
echo "\n------RANKER---------"
python grail_evaluate.py outputs/grailqa_v1.0_demo_reference.json misc/grail_demo_ranker_results.txt

echo "\n------FINAL---------"
python grail_evaluate.py outputs/grailqa_v1.0_demo_reference.json misc/grail_demo_final_results.txt


echo "FINAL results should be em: 0.843, f1: 0.847"