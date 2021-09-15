"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from .model_utils import get_inf_mask

from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
)
class RobertaForCandidateRanking(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sample_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert return_dict is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size,-1))
            token_type_ids = token_type_ids.view((batch_size * sample_size,-1)) if token_type_ids is not None else None
            attention_mask = attention_mask.view((batch_size * sample_size,-1))

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output[:, 0, :]) # take <s> token (equiv. to [CLS])
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.squeeze(1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
