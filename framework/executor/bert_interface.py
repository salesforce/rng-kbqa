"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from overrides import overrides
from pathlib import Path
from pytorch_transformers.modeling_auto import AutoModel
from pytorch_transformers.modeling_utils import PretrainedConfig
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import get_text_field_mask

path = str(Path(__file__).parent.absolute())


@TokenEmbedder.register("my_pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        config = PretrainedConfig.from_json_file(path + "/../bert_configs/debug.json")
        self.transformer_model = AutoModel.from_pretrained(model_name, config=config)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore
        attention_mask = get_text_field_mask({'bert': token_ids})
        # attention_mask = None
        # pylint: disable=arguments-differ
        token_type_ids = self.get_type_ids(token_ids)
        # position_ids = self.get_position_ids(token_ids).to(token_ids.device)
        # token_type_ids = None
        position_ids = None

        outputs = self.transformer_model(token_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                                         attention_mask=attention_mask)

        return outputs[0]

    def get_type_ids(self, token_ids: torch.LongTensor):
        type_ids = torch.zeros_like(token_ids)
        num_seq, max_len = token_ids.shape
        for i in range(num_seq):
            for j in range(max_len):
                if token_ids[i][j] == 102:  # id of [SEP], first occurence
                    break
            type_ids[i][j + 1:] = 1
        return type_ids

    def get_position_ids(self, token_ids: torch.LongTensor):
        position_ids = []
        num_seq, max_len = token_ids.shape
        for i in range(num_seq):
            position_ids_i = []
            next_id = 0
            # first_sep = True
            for j in range(max_len):
                position_ids_i.append(next_id)
                # if token_ids[i][j] == 102 and first_sep:
                if token_ids[i][j] == 102:
                    next_id = 0
                    # first_sep = False  # in case [SEP] is used as delimiter for schema constants
                else:
                    next_id += 1

            position_ids.append(position_ids_i)
        return torch.LongTensor(position_ids)

