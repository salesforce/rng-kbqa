"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import logging
from typing import Any, Dict, Optional, Tuple, Union
from numpy import core

import torch
from torch import mode, nn
from torch.utils.data import DistributedSampler, RandomSampler

from transformers import Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class GenerationTrainer(Trainer):

    # def compute_loss(self, model, inputs):
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs, use_cache=False)
    #     logits = outputs[0]
    #     return self._compute_loss(logits, labels, ignore_index=model.config.pad_token_id)

    # def _compute_loss(self, logits, labels, ignore_index):
    #     if self.args.label_smoothing == 0:
    #         # Same behavior as modeling_bart.py
    #         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    #         assert logits.shape[-1] == self.model.config.vocab_size
    #         loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    #     else:
    #         lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    #         loss, nll_loss = label_smoothed_nll_loss(
    #             lprobs, labels, self.args.label_smoothing, ignore_index=ignore_index
    #         )
    #     return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)
        core_model = model.module if isinstance(model, nn.DataParallel) else model
        max_length = (
            core_model.config.max_generate_length
            if hasattr(core_model.config, "max_generate_length")
            else core_model.config.max_position_embeddings
        )

        with torch.no_grad():
            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    num_beams=model.config.num_beams,
                    max_length=max_length,
                )
                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, max_length, core_model.config.pad_token_id
                )
            outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().item()
                labels_out = inputs.get("labels")
                logits = outputs[1]
            else:
                loss = None
                logits = outputs[0]
            logits = torch.argmax(logits, 2)
            logits = self._pad_tensors_to_max_len(logits, max_length, core_model.config.pad_token_id)
            if self.args.prediction_loss_only:
                logits = None
            else:
                logits = generated_tokens if self.args.predict_with_generate else logits

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels_out = labels_out.detach()
        labels = self._pad_tensors_to_max_len(labels_out, max_length, core_model.config.pad_token_id)
        # print(logits.size(), labels.size())
        # exit()
        return (loss, logits.detach(), labels)

    def _pad_tensors_to_max_len(self, tensor, max_length, pad_token_id):
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=max(self.args.warmup_steps, self.args.warmup_ratio * num_training_steps),
                num_training_steps=num_training_steps
            )
