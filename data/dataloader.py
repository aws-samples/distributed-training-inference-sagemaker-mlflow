#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torch.utils.data import DataLoader
from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.movielens import movielens_20m, COLUMN_TYPE_CASTERS
from typing import Any, Callable, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from math import ceil, floor
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
import torch


class MovielensDataset(Dataset):
    def __init__(self, csv_file, batch_size=None):
        self.data = pd.read_csv(csv_file, delimiter="::", names=DEFAULT_RATINGS_COLUMN_NAMES)
        self.batch_size = batch_size
        self.cur_idx = 0

    def __len__(self):
        return floor(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        users = []
        movies = []
        ratings = []
        for idx in range(self.batch_size):
            if self.cur_idx == len(self.data):
                break
            row = self.data.iloc[idx]
            user_id = row['userId']
            movie_id = row['movieId']
            rating_raw = row['rating']
            users.append(user_id)
            movies.append(movie_id)
            rating = 1 if rating_raw >= 4 else 0 
            ratings.append(rating)
            self.cur_idx += 1
        
        lengths = []
        values = []

        # For users
        length = torch.tensor( [1] * self.batch_size, dtype=torch.int32)
        value = torch.Tensor(users)
        lengths.append(length)
        values.append(value)

        # For items
        length = torch.tensor( [1] * self.batch_size, dtype=torch.int32)
        value = torch.Tensor(movies)
        lengths.append(length)
        values.append(value)

        sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=["userId", "movieId"],
            values=torch.cat(values),
            lengths=torch.cat(lengths),
        )

        dense_features = torch.Tensor([])

        labels = torch.Tensor(ratings)

        batch = Batch(
            dense_features=dense_features,
            sparse_features=sparse_features,
            labels=labels,
        )
        return batch

def get_dataloader(
    batch_size: int, num_embeddings: int, pin_memory: bool = False, num_workers: int = 0, csv_file: str = None
) -> DataLoader:
    """
    Gets a Random dataloader for the two tower model, containing a two_feature KJT as sparse_features, empty dense_features
    and binary labels

    Args:
        batch_size (int): batch_size
        num_embeddings (int): hash_size of the two embedding tables
        pin_memory (bool): Whether to pin_memory on the GPU
        num_workers (int) Number of dataloader workers
        csv_file (str) The movielens CSV file

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    return DataLoader(
        MovielensDataset(csv_file=csv_file, 
                          batch_size=batch_size),
    # return DataLoader(
    #     RandomRecDataset(
    #         keys=two_tower_column_names,
    #         batch_size=batch_size,
    #         hash_size=num_embeddings,
    #         ids_per_feature=1,
    #         num_dense=0,
    #     ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    
    