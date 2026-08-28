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

MEMORY FIX
-----------
Flattening every (node, timestep) row across ~120 Brake variants at
once produces a tensor too large to fit in RAM. So we stream through
each variant one at a time and write rows into a disk-backed .npy
memmap (build_pointwise_dataset_to_disk). RAM only ever holds one
variant's rows at a time.

DISK-SPEED FIX (important -- this replaces the earlier row-by-row
reader that caused training to effectively never finish on an
external HDD)
-------------------------------------------------------------------
The original PointwiseMemmapDataset read ONE ROW (28 bytes) at a time
in RANDOM order (because DataLoader(shuffle=True) picks random
indices). On a spinning/external HDD, every single row then requires
its own disk seek -- with ~412 million rows, this is catastrophically
slow (potentially days for one epoch).

ChunkedMemmapDataset instead:
  1. Reads large CONTIGUOUS chunks off disk (e.g. 200,000 rows at
     once) -- one sequential disk read per chunk, which HDDs handle
     far faster than scattered random reads.
  2. Shuffles rows WITHIN each chunk in memory (fast, RAM-only).
  3. Shuffles the ORDER of chunks each epoch, so training still sees
     data in a randomized order overall -- just not fully global
     row-level shuffling, which is an acceptable trade-off for the
     huge speed gain.

Requires: pip install torch numpy
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------
# Streaming dataset builder: writes flattened rows to disk as a memmap,
# never holding the full dataset in RAM.
# ---------------------------------------------------------------------
def build_pointwise_dataset_to_disk(data_list, out_dir, split_name, skip_if_exists=True):
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, f"{split_name}_inputs.npy")
    target_path = os.path.join(out_dir, f"{split_name}_targets.npy")
    meta_path = os.path.join(out_dir, f"{split_name}_meta.txt")

    if skip_if_exists and os.path.exists(input_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            total_rows = int(f.read().strip())
        print(f"[{split_name}] Found existing memmap on disk "
              f"({total_rows:,} rows) -- skipping rebuild.")
        return input_path, target_path, total_rows

    # --- Pass 1: figure out total row count ---
    total_rows = 0
    for data in data_list:
        n_nodes = data.x.shape[0]
        n_steps = data.y.shape[1]
        total_rows += n_nodes * n_steps

    print(f"[{split_name}] total rows = {total_rows:,} "
          f"(~{total_rows * 7 * 4 / 1e9:.2f} GB inputs, "
          f"~{total_rows * 1 * 4 / 1e9:.2f} GB targets on disk)")

    inputs_mm = np.lib.format.open_memmap(
        input_path, mode="w+", dtype=np.float32, shape=(total_rows, 7)
    )
    targets_mm = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(total_rows, 1)
    )

    # --- Pass 2: fill memmaps, one graph at a time ---
    row_ptr = 0
    for data in data_list:
        n_nodes = data.x.shape[0]
        n_steps = data.y.shape[1]

        static_rep = data.x.unsqueeze(1).expand(n_nodes, n_steps, 6)
        time_rep = data.time_stamps.view(1, n_steps, 1).expand(n_nodes, n_steps, 1)
        combined = torch.cat([static_rep, time_rep], dim=2).reshape(n_nodes * n_steps, 7)
        target = data.y.reshape(n_nodes * n_steps, 1)

        n_rows = combined.shape[0]
        inputs_mm[row_ptr:row_ptr + n_rows] = combined.numpy().astype(np.float32)
        targets_mm[row_ptr:row_ptr + n_rows] = target.numpy().astype(np.float32)
        row_ptr += n_rows

    inputs_mm.flush()
    targets_mm.flush()
    del inputs_mm, targets_mm

    with open(meta_path, "w") as f:
        f.write(str(total_rows))

    return input_path, target_path, total_rows


# ---------------------------------------------------------------------
# Streaming normalization stats: running mean/std, one graph at a time.
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Chunked, disk-friendly dataset -- see module docstring above for why
# this replaced the old row-by-row PointwiseMemmapDataset.
# ---------------------------------------------------------------------
class ChunkedMemmapDataset(IterableDataset):
    def __init__(self, input_path, target_path, total_rows, stats, chunk_size=200_000):
        self.input_path = input_path
        self.target_path = target_path
        self.total_rows = total_rows
        self.stats = stats
        self.chunk_size = chunk_size

    def __len__(self):
        return self.total_rows

    def __iter__(self):
        inputs = np.load(self.input_path, mmap_mode="r")
        targets = np.load(self.target_path, mmap_mode="r")

        n_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size
        chunk_order = np.random.permutation(n_chunks)

        for c in chunk_order:
            start = int(c) * self.chunk_size
            end = min(start + self.chunk_size, self.total_rows)

            # One sequential disk read per chunk -- fast even on HDD
            x_chunk = torch.from_numpy(np.array(inputs[start:end])).float()
            y_chunk = torch.from_numpy(np.array(targets[start:end])).float()

            x_chunk = (x_chunk - self.stats["input_mean"]) / self.stats["input_std"]
            y_chunk = (y_chunk - self.stats["target_mean"]) / self.stats["target_std"]

            perm = torch.randperm(x_chunk.shape[0])
            x_chunk, y_chunk = x_chunk[perm], y_chunk[perm]

            for i in range(x_chunk.shape[0]):
                yield x_chunk[i], y_chunk[i]


def denormalize_targets(y_n, stats):
    return y_n * stats["target_std"] + stats["target_mean"]


class TemperatureMLP(nn.Module):
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