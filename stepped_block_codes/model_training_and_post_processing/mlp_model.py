"""
mlp_model.py

WHAT IS AN MLP (Multi-Layer Perceptron)
-----------------------------------------
The simplest neural network: a stack of fully-connected ("Linear")
layers with a nonlinearity (ReLU) between them. It has NO built-in
notion of mesh connectivity, neighbors, or time -- every input row is
treated as an independent sample.

To predict a temperature FIELD over TIME with an MLP, we must:
  1. Treat every (node, timestep) pair as its own independent row.
  2. Explicitly add physical time as an input feature.

  input  = [X, Y, Z, SDF, Pour_Temp, Mould_Temp, time_stamp]  (7 features)
  target = Temperature at that node, at that time              (1 value)

DATA PIPELINE
--------------
OnTheFlyBatchDataset avoids two earlier failure modes:
  1. Pre-flattening the WHOLE dataset to a disk memmap -- this blew
     past Kaggle's session disk limit for large geometries.
  2. Yielding one row at a time in Python -- this caused a severe
     CPU-bound bottleneck (billions of individual `yield` calls),
     leaving the GPU mostly idle.

Instead: for each epoch, variant order is shuffled, and for EACH
variant (one at a time, never all variants in memory simultaneously
as one giant tensor), its [N*T, 7] pointwise rows are computed in a
few vectorized tensor ops, shuffled internally, then sliced into
batch-sized chunks and yielded directly as ready-made batches --
no per-row Python loop, no disk writes.

This file is geometry-agnostic -- the same mlp_model.py is reused for
Brake, Flange, and Stepped_Block. Only train_mlp.py's CONFIG block
(paths, geometry name) changes between geometries.

Requires: pip install torch numpy
"""

import numpy as np
import torch
import torch.nn as nn


def compute_norm_stats_streaming(data_list):
    input_sum = torch.zeros(7, dtype=torch.float64)
    input_sumsq = torch.zeros(7, dtype=torch.float64)
    target_sum = torch.zeros(1, dtype=torch.float64)
    target_sumsq = torch.zeros(1, dtype=torch.float64)
    total_rows = 0

    for data in data_list:
        n_nodes = data.x.shape[0]
        n_steps = data.y.shape[1]

        static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
        time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
        combined = torch.cat([static_rep, time_rep], dim=2).reshape(-1, 7).double()
        target = data.y.reshape(-1, 1).double()

        input_sum += combined.sum(dim=0)
        input_sumsq += (combined ** 2).sum(dim=0)
        target_sum += target.sum(dim=0)
        target_sumsq += (target ** 2).sum(dim=0)
        total_rows += combined.shape[0]

    input_mean = (input_sum / total_rows).float()
    input_var = (input_sumsq / total_rows).float() - input_mean ** 2
    input_std = input_var.clamp_min(1e-8).sqrt()

    target_mean = (target_sum / total_rows).float()
    target_var = (target_sumsq / total_rows).float() - target_mean ** 2
    target_std = target_var.clamp_min(1e-8).sqrt()

    return {
        "input_mean": input_mean.view(1, -1),
        "input_std": input_std.view(1, -1),
        "target_mean": target_mean.view(1, -1),
        "target_std": target_std.view(1, -1),
    }


class OnTheFlyBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_list, stats, batch_size):
        self.data_list = data_list
        self.stats = stats
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_data_list = self.data_list
        else:
            my_data_list = self.data_list[worker_info.id::worker_info.num_workers]

        order = np.random.permutation(len(my_data_list))
        for idx in order:
            data = my_data_list[idx]
            n_nodes = data.x.shape[0]
            n_steps = data.y.shape[1]

            static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
            time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
            x = torch.cat([static_rep, time_rep], dim=2).reshape(-1, 7)
            y = data.y.reshape(-1, 1)

            x_n = (x - self.stats["input_mean"]) / self.stats["input_std"]
            y_n = (y - self.stats["target_mean"]) / self.stats["target_std"]

            perm = torch.randperm(x_n.shape[0])
            x_n, y_n = x_n[perm], y_n[perm]

            n_rows = x_n.shape[0]
            for start in range(0, n_rows, self.batch_size):
                end = min(start + self.batch_size, n_rows)
                yield x_n[start:end], y_n[start:end]


def denormalize_targets(y_n, stats):
    return y_n * stats["target_std"] + stats["target_mean"]


class TemperatureMLP(nn.Module):
    """
    4-layer feedforward MLP.
    Input:  [batch, 7]  -> X, Y, Z, SDF, Pour_Temp, Mould_Temp, time_s (normalized)
    Output: [batch, 1]  -> predicted temperature (normalized)
    Total trainable parameters: 25,857
    """
    def __init__(self, in_dim=7, hidden_dim=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)