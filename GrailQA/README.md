## Training Model on GrailQA
Please read the instructions in the root folder first.

### Train Entity Disambiguator

Prepare entity mention training/dev data

```
python detect_entity_mentions.py --split train
python detect_entity_mentions.py --split dev
```

Train the disambiguation model.

 **!!** We use a batch size of 16, please modify the `per_gpu_train_batch_size` and `gradient_accumulation_steps` in the script `scripts/run_disamb.sh ` accordingly to make the total batch size (`num_gpu * per_gpu_train_batch_size * gradient_accumulation_steps`) as 16.

`sh scripts/run_disamb.sh train <exp_id>`, where the `<exp_id>` is the ID you name. The training outputs will be stored at `exps/disamb_grail_<exp_id>`.

After traininig the model, please use the model to run entity disambiguation for the `dev` set as in the `README` of the root folder.

### Train Ranker

Prepare ranking candidates file

```
python enumerate_candidates.py --split train # we use gt entity for trainning (so no need for prediction on training)
python enumerate_candidates.py --split dev --pred_file misc/grail_dev_entity_linking.json
```


Train the ranker model (**!!** We use a batch size of 8, please modify the script accordingly).

`sh scripts/run_ranker.sh train  <exp_id>`. The training outputs will be stored at `exps/rank_grail_<exp_id>`.


After training the model, please use the trained model to obtain candidate logits for both `train` and `dev`.


### Train Generator

Prepare generation data file

```
python make_generation_dataset.py --split train --logit_file misc/grail_train_candidates_logits.bin
python make_generation_dataset.py --split dev --logit_file misc/grail_dev_candidates_logits.bin
```


Train the generator model (**!!** We use a batch size of 8, please modify the script accordingly).

`sh scripts/run_gen.sh train  <exp_id>`. The training outputs will be stored at `exps/gen_grail_<exp_id>`.
