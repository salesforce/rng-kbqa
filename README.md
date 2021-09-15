# RnG-KBQA

Code for RnG-KBQA: Rank-and-Generate Approach for Question Answering over Knowledge Base.


## Requirements

The code is tested under the following environment setup

* python==3.8.10
* pytorch==1.7.0
* transformers==3.3.1
* spacy==3.1.1
* other requirments please see `requirements.txt`

#### System requirements:

It's recommended to use a machine with over 300G memory to train the models, and use a machine with 128G memory for inference. However, 256G memory will still be sufficient for runing inference and training all of the models (some tricks for saving memorry is needed in training ranker model for GrailQA).

## General Setup

**Setup Experiment Directory**

Before Running the scripts, please use the `setup.sh` to setup the experiment folder. Basically it creates some symbolic links in each exp directory.

**Setup Freebase** 

All of the datasets use Freebase as the knowledge source. Please follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso triplestore service. If you modify the default url, you may need to change the url in `/framework/executor/sparql_executor.py` accordingly, after starting your virtuoso service, 

## Reproducing the Results on GrailQA

Please use `/GrailQA` as the working directory when running experiments on GrailQA.

---

#### Prepare dataset and pretrained checkpoints

**Dataset**

Please download the dataset and put the them under `outputs` so as to organize dataset as `outputs/grailqa_v1.0_train/dev/test.json`. (Please rename `test-public` split to `test` split).

**NER Checkpoints**

We use the NER system (under directory `entity_linking` and `entity_linker`) from [Original GrailQA](https://github.com/dki-lab/GrailQA) Code Repo. Please use the following instructions (copied from oringinal repo) to pull related data

* Download the mentions data from https://1drv.ms/u/s!AuJiG47gLqTznjl7VbnOESK6qPW2?e=HDy2Ye and put all data under `entity_linker/data/`.
* Download our trained NER model from https://1drv.ms/u/s!AuJiG47gLqTznjge7wLqAZiSMIcU?e=5RpKaC, which is trained using the training data of GrailQA, and put it under `entity_linker/BERT_NER/`.

**Other Checkpoints**

Please download the following checkpoints for entity disambiguation, candidate ranking, and augmented generation checkpoints, unzip and put them under `checkpoints/` directory

* Entity Disambiguation: [grail_bert_entity_disamb](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/model_release/grail_bert_entity_disamb.zip) 
* Candidate Ranking: [grail_bert_ranking](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/model_release/grail_bert_ranking.zip) 
* Augmented Generation: [grail_t5_generation](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/model_release/grail_t5_generation.zip) 

**KB Cache**

We attach the cache of query results from KB, which can help save some time. Please download the [cache file for grailqa](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/KB_cache/grail.zip), unzip and put them under `cache/`, so that we have `cache/grail-LinkedRelation.bin` and `cache/grail-TwoHopPath.bin` in the place.

---

#### Running inference

## Demo for Checking the Pipeline

It's recommended to use the one-click demo scripts first to test if everything mentioned above is setup correctly. If it successfully run through, you'll get a final F1 of 0.847. Please make sure you successfully reproduce the results on this small demo set first, as inference on `dev` and `test` can take a long time.

`sh scripts/walk_through_demo.sh`

## Step by Step Instructions

We also provide step-by-step inference instructions as below:

**(i)  Detecting Entities**

Once having the entity linker ready, run

`python detect_entity_mention.py --split <split> # eg. --split test`

This will write entity mentions to `outputs/grail_<split>_entities.json`, we extract up to 10 entities for each mention, which will be further disambiguate in the next step.

!! Running entity detection for the first time will require building surface form index, which can take a long time (but it's only needed for the first time).

**(ii) Disambiguating Entities (Entity Linking)**

We have provided pretrained ranker model

`sh scripts/run_disamb.sh predict <model_path> <split>`

E.g., `sh scripts/run_disamb.sh predict checkpoints/grail_bert_entity_disamb test`

This will write the prediction results (in the form of selected entity index for each mention) to `misc/grail_<split>_entity_linking.json`.

**(iii) Enumerating Logical Form Candidates**

`python enumerate_candidates.py --split <split> --pred_file <pred_file>`

E.g., `python enumerate_candidates.py --split test --pred_file misc/grail_test_entity_linking.json`.

This will write enumerated candidates to `outputs/grail_<split>_candidates-ranking.jsonline`.

**(iv) Running Ranker**

`sh scripts/run_ranker.sh predict <model_path> <split>`

E.g., `sh scripts/run_ranker.sh predict checkpoints/grail_bert_ranking test`

This will write prediction candidate logits (the logits of each candidate for each example) to `misc/grail_<split>_candidates_logits.bin`, and prediction result (in original GrailQA prediction format) to `misc/grail_<split>_ranker_results.txt`

You may evaluate the ranker results by `python grail_evaluate.py <path_to_data_split> <path_to_predictions>`

E.g., `python grail_evaluate.py outputs/grailqa_v1.0_dev.json misc/grail_dev_ranker_results.txt`

**(v) Running Generator**

First, make prepare generation model inputs

`python make_generation_dataset.py --split <split> --logit_file <pred_file>`

E.g., `python make_generation_dataset.py --split test --logit_file misc/grail_test_candidate_logits.bin`.

This will read the canddiates and the use logits to select top-k candidates and write generation model inputs to `outputs/grail_<split>_gen.json`.

Second, run generation model to get the top-k prediction

`sh scripts/run_gen.sh predict <model_path> <split>`

E.g., `sh scripts/run_gen.sh predict checkpoints/grail_t5_generation test`.

This will generate top-k decoded logical forms stored at `misc/grail_<split>_topk_generations.json`.

**(vi) Final Inference Steps**

Having the decoded top-k predictions, we'll go down the top-k list, execute the logical form one by one until we find one logical form return valid answers.

`python eval_topk_prediction.py --split <split> --pred_file <pred_file>`

E.g., `python eval_topk_prediction.py --split test --pred_file misc/grail_test_topk_generations.json`

prediction result (in original GrailQA prediction format) to `misc/grail_<split>_final_results.txt`.

You can then use official GrailQA evaluate script to run evaluation

`python grail_evaluate.py <path_to_data_split> <path_to_predictions>`

E.g., `python grail_evaluate.py outputs/grailqa_v1.0_dev.json misc/grail_dev_final_results.txt`

---

#### Training Models
We already attached pretrained-models ready for running inference. If you'd like to train your own models please checkout the `README` at `/GrailQA` folder.


## Reproducing the Results on WebQSP


Please use `/WebQSP` as the working directory when running experiments on WebQSP.

---

#### Prepare dataset and pretrained checkpoints

**Dataset**

Please download the [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) dataset and put the them under `outputs` so as to organize dataset as `outputs/WebQSP.train[test].json`.


**Model Checkpoints**

Please download the following checkpoints for candidate ranking, and augmented generation checkpoints, unzip and put them under `checkpoints/` directory

* Candidate Ranking: [webqsp_bert_ranking](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/model_release/webqsp_bert_ranking.zip) 
* Augmented Generation: [webqsp_t5_generation](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/model_release/webqsp_t5_generation.zip) 

**KB Cache**

 Please download the [cache file for webqsp](https://storage.cloud.google.com/sfr-rng-kbqa-data-research/KB_cache/webqsp.zip), unzip and put them under `cache/` so that we have `cache/webqsp-LinkedRelation.bin` and `cache/webqsp-TwoHopPath.bin` in the place.

---

#### Running inference

**(i) Parsing Sparql-Query to S-Expression**

<!-- This step can be ***skipped***, as we've already include outputs of this step (`outputs/WebQSP.ptrain.expr.json`, `outputs/WebQSP.pdev.expr.json`, `outputs/WebQSP.test.expr.json`). We split the original `train` into `ptrain` `pdev` for validation purpose. -->

As stated in the paper, we generate s-expressions, which is not provided by the original dataset, so we provide scripts to parse sparql-query to s-expressions.

Run `python parse_sparql.py`, which will augment original dataset files with s-expressions and save them in `outputs` as `outputs/WebQSP.train.expr.json` and `outputs/WebQSP.dev.expr.json`. Since there is no validation set, we further randomly select 200 examples from the training set for validation, yielding `ptrain` split and `pdev` split.

**(ii)  Entity Detection and Linking using ELQ**

This step can be ***skipped***, as we've already include outputs of this step (`misc/webqsp_train_elq-5_mid.json`, `outputs/webqsp_test_elq-5_mid.json`).

The scripts and config of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) model can be found in `elq_linking/run_elq_linker.py`. If you'd like to use the script to run entity linking, please copy the `run_elq_linker.py` python script to ELQ model and run the script there. 


**(iii) Enumerating Logical Form Candidates**

`python enumerate_candidates.py --split test`

This will write enumerated candidates to `outputs/webqsp_test_candidates-ranking.jsonline`.

**(iv) Runing Ranker**

`sh scripts/run_ranker.sh predict checkpoints/webqsp_bert_ranking test`

This will write prediction candidate logits (the logits of each candidate for each example) to `misc/webqsp_test_candidates_logits.bin`, and prediction result (in original GrailQA prediction format) to `misc/webqsp_test_ranker_results.txt`

**(v) Running Generator**

First, make prepare generation model inputs

`python make_generation_dataset.py --split test --logit_file misc/webqsp_test_candidate_logits.bin`.

This will read the candidates and the use logits to select top-k candidates and write generation model inputs to `outputs/webqsp_test_gen.json`.

Second, run generation model to get the top-k prediction

`sh scripts/run_gen.sh predict checkpoints/webqsp_t5_generation test`

This will generate top-k decoded logical forms stored at `misc/webqsp_test_topk_generations.json`.

**(vi) Final Inference Steps**

Having the decoded top-k predictions, we'll go down the top-k list, execute the logical form one by one until we find one logical form return valid answers.

`python eval_topk_prediction.py --split test --pred_file misc/wepqsp_test_topk_generations.json`

Prediction result will be stored (in GrailQA prediction format) to `misc/webqsp_test_final_results.txt`.

You can then use official WebQSP (only modified in I/O) evaluate script to run evaluation

`python webqsp_evaluate.py outputs/WebQSP.test.json misc/webqsp_test_final_results.txt`.

---

#### Training Models
We already attached pretrained-models ready for running inference. If you'd like to train your own models please checkout the `README` at `/WebQSP` folder.


## Questions?
For any questions, feel free to open issues, or shoot emails to
- Semih Yavuz (syavuz@salesforce.com)
- [Xi Ye](https://www.cs.utexas.edu/~xiye/)

## License
The code is released under BSD 3-Clause - see [LICENSE](LICENSE.txt) for details.
