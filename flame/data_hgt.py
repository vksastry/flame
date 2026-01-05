# data_hgt.py 

import os
import glob
from typing import Sequence, Optional, Union

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from flame.config_manager import JobConfig
from torchtitan.tools.logging import init_logger, logger
import pdb

class HGTSpatioTemporalDataset(Dataset):
    """
    Simple spatio-temporal dataset for preprocessed NCEP HGT data.
    Expects a single .npy file with shape:
    (T, C, H, W)  = (time, levels, lat, lon)
    Creates sliding windows:
    inputs:  (T_in, C_sel, H, W)
    targets: (T_out, C_sel, H, W)
    """
    def __init__(
            self,
            data_files: str,
            input_len: int,
            target_len: int,
            stride: int = 1,
            level: Optional[Union[int, Sequence[int]]] = None,
    ):
        """
        Parameters
        ----------
        npy_path : str
            Path to the .npy file with shape (T, C, H, W), dtype float32.
        input_len : int
            Number of input timesteps (T_in).
        target_len : int
            Number of target timesteps (T_out).
        stride : int
            Stride between window start indices.
        level : int or list[int] or None
            Which channel(s) to keep along the C dimension.
            - If int: single level index (kept as C=1).
            - If list/tuple: multiple indices.
            - If None: keep all C levels.
        """
        super().__init__()

        # Memory-map the big array so we don't load 80GB into RAM at once
        arr = np.load(data_files, mmap_mode="r")  # (T, C, H, W) memmap or ndarray
        print("done read")
        if arr.ndim != 4:
            raise ValueError(f"Expected array with 4 dims (T,C,H,W), got {arr.shape}")
        # Select channels/levels if requested
        if level is not None:
            if isinstance(level, int):
                # Keep a single channel but preserve a C dimension of size 1
                arr = arr[:, level:level + 1, :, :]
            else:
                # Assume it's a sequence of ints
                level_idx = np.array(level, dtype=int)
                arr = arr[:, level_idx, :, :]

        self.data = arr  # (T, C_sel, H, W), float32, memmap-backed
        #pdb.set_trace()
        #data_flat = self.data.reshape(-1)
        #mean: 11884.07 std: 9119.372
        self.mean = 11884.07 #self.data.mean()
        self.std = 9119.372 #self.data.std()
        print("Normalization stats:")
        print(" mean:", self.mean)
        print(" std :", self.std)
        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride

        self.T_total, self.C, self.H, self.W = self.data.shape

        # How many windows we can extract
        self.max_start = self.T_total - (input_len + target_len)
        if self.max_start < 0:
            raise ValueError(
                    f"Not enough time steps ({self.T_total}) for "
                    f"input_len={input_len} + target_len={target_len}"
            )
        
    def __len__(self) -> int:
        # number of windows with given stride
        return 1 + self.max_start // self.stride
         
    def __getitem__(self, idx: int):
        t0 = idx * self.stride
        t1 = t0 + self.input_len
        t2 = t1 + self.target_len

        # Slices of memmap → views, cheap
        x_np = self.data[t0:t1]  # (T_in, C, H, W)
        y_np = self.data[t1:t2]  # (T_out, C, H, W)

        # normalize the data 
        x_np = (x_np - self.mean) / self.std
        y_np = (y_np - self.mean) / self.std

        # np.asarray keeps it as view, torch.from_numpy can wrap memmap-backed arrays
        x = torch.from_numpy(np.asarray(x_np))
        y = torch.from_numpy(np.asarray(y_np))

        return {"inputs": x, "targets": y}

def build_hgt_dataloader(
    job_config,
    rank: int,
    world_size: int,
):
    """
    Build a DataLoader for the NCEP HGT dataset.

    Expected config fields (you can rename as you like):

        training.hgt_data_dir    : directory with yearly hgt*.nc files
        training.input_len       : input sequence length (T_in)
        training.target_len      : target sequence length (T_out)
        training.stride          : stride in time
        training.levels          : list or scalar, e.g. 500.0
        training.years           : list of years or empty => all
        training.batch_size
        training.num_workers
        training.pin_memory
        training.persistent_workers
    """
    data_files = job_config.training.data_files

    input_len = getattr(job_config.training, "input_len", 6)
    target_len = getattr(job_config.training, "target_len", 1)
    stride = getattr(job_config.training, "stride", 1)

    # optional; if not present, default to 500 hPa
    levels = getattr(job_config.training, "levels", 7)
    years = getattr(job_config.training, "years", None)
    
    print(f"get HGTSpatioTemporalDataset")
    dataset = HGTSpatioTemporalDataset(
        data_files=data_files,
        input_len=input_len,
        target_len=target_len,
        stride=stride,
        level=int(levels),
    )
    
    print(f"get the dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=job_config.training.batch_size,
        shuffle=True,
        num_workers=job_config.training.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return dataloader

# ---------------------------------------------------------------------
# Main: simple test of the dataloader
# ---------------------------------------------------------------------
def main():
    init_logger()

    job_config = JobConfig()
    job_config.parse_args()

    # For a simple test we treat this as single-process, rank 0.
    rank = 0
    world_size = 1

    loader = build_hgt_dataloader(job_config, rank, world_size)
    
    logger.info("Iterating over a few batches from HGT dataloader...")
    for i, batch in enumerate(loader):
        x = batch["inputs"]   # (B, T_in, C, H, W)
        y = batch["targets"]  # (B, T_out, C, H, W)

        logger.info(
            f"Batch {i}: inputs.shape={tuple(x.shape)}, "
            f"targets.shape={tuple(y.shape)}"
        )
        if i >= 2:
            break

    logger.info("HGT dataloader test done.")


if __name__ == "__main__":
    main()
