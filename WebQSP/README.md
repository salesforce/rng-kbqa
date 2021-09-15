## Training Model on WebQSP
Please read the instructions in the root folder first.

### Train Ranker

Prepare ranking candidates file

`python enumerate_candidates.py --split traindev`, which will write candidates file needed for training at `outputs/webqsp_ptrain(pdev)_candidates-ranking.jsonline`.

Train the ranker model

 **!!** We use a batch size of 8, please modify the `per_gpu_train_batch_size` and `gradient_accumulation_steps` in the script `scripts/run_ranker.sh ` accordingly to make the total batch size (`num_gpu * per_gpu_train_batch_size * gradient_accumulation_steps`) as 8.

`sh scripts/run_ranker.sh train <exp_id>`. The training outputs will be stored at `exps/rank_webqsp_<exp_id>`.

After training the model, please use the trained model to obtain candidate logits for both `ptrain` and `pdev` as in the `README` of the root folder.

### Train Generator

Prepare generation data file

```
python make_generation_dataset.py --split train --logit_file misc/grail_train_candidates_logits.bin
python make_generation_dataset.py --split dev --logit_file misc/grail_dev_candidates_logits.bin
```


Train the generator model (**!!** We use a batch size of 8, please modify the script accordingly).

`sh scripts/run_gen.sh train  <exp_id>`. The training outputs will be stored at `exps/gen_grail_<exp_id>`.
