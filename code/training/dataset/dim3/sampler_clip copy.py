#pedro
import math
import random
import torch
from torch.utils.data import Sampler
import pandas as pd


from collections import Counter
import math

def redistribute_ids(id_list, id_dict, error_margin=0.2):
    """
    1) Count how many times each ID appears in id_list (list_count).
    2) Count how many times each ID appears across all lists in id_dict (df_count).
    3) Compute count_ratio = list_count / df_count for each ID.
    4) Build a new dict where each sublist is “stretched” by duplicating each ID
       round(count_ratio) times (but at least 1).
    5) Sanity‐check that the merged counts of the new dict match id_list counts
       within error_margin (default 10%).

    Returns:
        new_dict: a dict with the same keys as id_dict but new (expanded) lists.
    """
    # 1) original counts
    list_count = Counter(id_list)

    # 2) counts in the “dataframe” dict
    df_count = Counter()
    for sub in id_dict.values():
        df_count.update(sub)

    # 3) compute ratios (default to 1 if an id never appeared in df_count)
    ratios = {
        id_: list_count[id_] / df_count.get(id_, 1)
        for id_ in list_count
    }

    # 4) build the expanded dict
    new_dict = {}
    for key, sub in id_dict.items():
        expanded = []
        for id_ in sub:
            ratio = ratios.get(id_, 1)
            # duplicate count = round(ratio), but at least 1
            n = max(1, int(round(ratio)))
            expanded.extend([id_] * n)
        new_dict[key] = expanded

    # 5) sanity‐check
    merged = []
    for expanded in new_dict.values():
        merged.extend(expanded)
    new_count = Counter(merged)

    for id_, orig in list_count.items():
        new_c = new_count.get(id_, 0)
        if abs(new_c - orig) > error_margin * orig:
            raise ValueError(
                f"ID {id_}: original={orig}, rebuilt={new_c} "
                f"(>{error_margin*100:.0f}% difference)"
            )

    return new_dict

import math, random
from torch.utils.data import BatchSampler

import math
import random
from torch.utils.data import BatchSampler
class SequentialBalancedChunkedBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 samples_per_epoch: int,
                 batch_size: int,
                 shuffle: bool = True,
                 seed: int = 0,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Args:
            dataset: your Dataset with .img_list & .tumors_per_type
            samples_per_epoch: how many samples/epoch you want
            batch_size: total size of each batch (will be split 1/3,1/3,1/3+rem)
            shuffle: whether to reshuffle once per full pass
            seed: random seed
        """

        # 1) rebuild the ID→index pools exactly as before
        itens = dataset.img_list
        self.itens = [
            item[item.find('BDMAP_'):item.find('BDMAP_') + len('BDMAP_00001111')]
            for item in itens
        ]
        self.crop_registry = dataset.crop_registry
        dataset.tumors_per_type = 
        self.tumors_per_type = redistribute_ids(self.itens,
                                                dataset.tumors_per_type,
                                                error_margin=0.2)
        idxs_per_organ = {
            organ: [self.itens.index(bd) for bd in id_list]
            for organ, id_list in self.tumors_per_type.items()
        }
        self.healthy = idxs_per_organ.pop('healthy', [])
        self.idxs_per_organ = idxs_per_organ
        self.organs        = list(idxs_per_organ)

        # 2) derive per‐group sizes from batch_size
        self.batch_size = batch_size
        self.n_each     = batch_size // 3           # organ & healthy
        self.n_other    = batch_size - 2*self.n_each # “other” gets remainder

        # 3) cycle/chunk logic
        self.dataset_size      = len(self.itens)
        self.samples_per_epoch = samples_per_epoch
        self.cycle_length      = math.ceil(self.dataset_size / samples_per_epoch)
        self.batches_per_epoch = math.ceil(samples_per_epoch / batch_size)

        self.shuffle = shuffle
        self.seed    = seed
        self.epoch   = 0
        self.cycle   = -1
        self.shuffled_indices = list(range(self.dataset_size))
        
        # store DDP info
        self.rank       = rank
        self.world_size = world_size


        super().__init__(sampler=None,
                         batch_size=self.batch_size,
                         drop_last=False)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # reshuffle once per cycle
        new_cycle = self.epoch // self.cycle_length
        if new_cycle != self.cycle:
            self.cycle = new_cycle
            rnd = random.Random(self.seed + self.cycle)

            rnd.shuffle(self.shuffled_indices)
            for lst in self.idxs_per_organ.values():
                rnd.shuffle(lst)
            rnd.shuffle(self.healthy)
            rnd.shuffle(self.organs)

        # carve out this epoch’s chunk
        within = self.epoch % self.cycle_length
        start  = within * self.samples_per_epoch
        end    = start + self.samples_per_epoch
        chunk  = self.shuffled_indices[start : min(end, self.dataset_size)]

        # pad last chunk if short
        short = self.samples_per_epoch - len(chunk)
        if short > 0:
            pool = (self.shuffled_indices[:start] +
                    self.shuffled_indices[end:])
            if not pool:
                pool = self.shuffled_indices
            chunk += random.Random(self.seed + self.cycle)\
                         .choices(pool, k=short)

        chunk_set = set(chunk)

        # restrict organ/healthy to the chunk
        local_org     = {
            organ: [i for i in lst if i in chunk_set]
            for organ, lst in self.idxs_per_organ.items()
        }
        local_healthy = [i for i in self.healthy if i in chunk_set]

        # build other‐pool per organ & shuffle once
        local_other = {
            organ: [
                i
                for o, lst in local_org.items() if o != organ
                for i in lst
            ]
            for organ in self.organs
        }
        if self.shuffle:
            rnd2 = random.Random(self.seed + self.cycle + self.epoch + 12345)
            for lst in local_other.values():
                rnd2.shuffle(lst)

        # helper: sequential slice with wrap‑around
        def slice_seq(pool, batch_idx, n):
            L = len(pool)
            if L == 0:
                return []
            s = (batch_idx * n) % L
            if s + n <= L:
                return pool[s:s+n]
            first = pool[s:]
            rem   = n - len(first)
            loops = rem // L
            extra = rem % L
            return first + pool*loops + pool[:extra]

        # emit each batch
        for b in range(self.batches_per_epoch):
            organ     = self.organs[b % len(self.organs)]
            org_pool   = local_org.get(organ, [])
            other_pool = local_other.get(organ, [])

            # global batch of size `batch_size`
            global_batch = (
                slice_seq(org_pool,      b, self.n_each) +
                slice_seq(local_healthy, b, self.n_each) +
                slice_seq(other_pool,    b, self.n_other)
            )

            # now slice per-rank, round-robin style
            local_batch = global_batch[self.rank :: self.world_size]
            yield local_batch

    def __len__(self):
        # how many LOCAL batches this rank will see
        return math.ceil(self.batches_per_epoch / self.world_size)





