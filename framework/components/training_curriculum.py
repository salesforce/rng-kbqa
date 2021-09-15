"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from typing import overload
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from components.rank_dataset import contrastive_collate_fn

class TrainingCurriculum:
    def __init__(self, args, train_dataset, tokenizer):
        self.random_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        self.random_collate_fn = partial(contrastive_collate_fn, tokenizer=tokenizer, num_sample=args.num_contrast_sample, strategy='random')
        self.train_batch_size = args.train_batch_size

    def get_initial_dataloader(self, dataset):
        return DataLoader(dataset, sampler=self.random_sampler, batch_size=self.train_batch_size, collate_fn=self.random_collate_fn)

    def get_action_flag_and_dataloader_for_epoch(self, dataset, epoch):
        raise NotImplementedError('Abstract curriculum type')
    
    def summary(self):
        raise NotImplementedError('Abstract curriculum type')

    @classmethod
    def from_args(cls, args, dataset, tokenizer):
        if args.training_curriculum == 'random':
            return RandomCurriculum(args, dataset, tokenizer)
        elif args.training_curriculum == 'bootstrap':
            return BootstrCurriculum(args, dataset, tokenizer)
        elif args.training_curriculum == 'mixbootstrap':
            return MixBootstrCurriculum(args, dataset, tokenizer)
        else:
            raise RuntimeError('Curriculum type not supported')

class RandomCurriculum(TrainingCurriculum):
    def get_action_flag_and_dataloader_for_epoch(self, dataset, epoch):
        return False, DataLoader(dataset, sampler=self.random_sampler, batch_size=self.train_batch_size, collate_fn=self.random_collate_fn)

    def summary(self):
        return 'completely random curriculum'

# bootstrapping, after some epoch
class BootstrCurriculum(TrainingCurriculum):
    def __init__(self, args, dataset, tokenizer):
        super().__init__(args, dataset, tokenizer)
        self.bs_start = args.bootstrapping_start
        self.bs_update_epochs = args.bootstrapping_update_epochs
        self.advanced_collate_fn = partial(contrastive_collate_fn, tokenizer=tokenizer, num_sample=args.num_contrast_sample, strategy='boostrap')
    
    def get_action_flag_and_dataloader_for_epoch(self, dataset, epoch):
        flag = epoch in self.bs_update_epochs
        seletec_fn = self.advanced_collate_fn if epoch >= self.bs_start else self.random_collate_fn
        return flag, DataLoader(dataset, sampler=self.random_sampler, batch_size=self.train_batch_size, collate_fn=seletec_fn)
    
    def summary(self):
        return 'bootstrapping strategy, starting from {}, ticks at {}'.format(self.bs_start, '[' + ','.join(map(str,self.bs_update_epochs)) + ']')

# 50% bootstrapping + 50% random, after some specific epoch
class MixBootstrCurriculum(TrainingCurriculum):
    def __init__(self, args, dataset, tokenizer):
        super().__init__(args, dataset, tokenizer)
        self.bs_start = args.bootstrapping_start
        self.bs_update_epochs = args.bootstrapping_update_epochs
        self.advanced_collate_fn = partial(contrastive_collate_fn, tokenizer=tokenizer, num_sample=args.num_contrast_sample, strategy='mixboostrap')
    
    def get_action_flag_and_dataloader_for_epoch(self, dataset, epoch):
        flag = epoch in self.bs_update_epochs
        seletec_fn = self.advanced_collate_fn if epoch >= self.bs_start else self.random_collate_fn
        return flag, DataLoader(dataset, sampler=self.random_sampler, batch_size=self.train_batch_size, collate_fn=seletec_fn)

    def summary(self):
        return 'mixbootstrapping strategy, starting from {}, ticks at {}'.format(self.bs_start, '[' + ','.join(map(str,self.bs_update_epochs)) + ']')
