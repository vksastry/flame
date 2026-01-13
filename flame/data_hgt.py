# data_hgt.py 

import os
import glob
from typing import Sequence, Optional, Union

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from flame.config_manager import JobConfig
from torchtitan.tools.logging import init_logger, logger
import pdb

means = np.array([   89.57868,   726.8976 ,  1410.4211 ,  2948.8853 ,  4140.5693 ,
        5509.7495 ,  7128.5396 ,  9111.014  , 10317.388  , 11761.315  ,
       13581.902  , 16087.761  , 18275.088  , 20361.256  , 23576.166  ,
       26173.098  , 30699.953  ], dtype=float)
stds = np.array([ 107.536415,  117.37903 ,  140.21944 ,  213.68188 ,  275.05136 ,
        343.34845 ,  425.6921  ,  520.3599  ,  564.0767  ,  595.05    ,
        602.211   ,  575.09674 ,  573.84155 ,  614.817   ,  726.8138  ,
        835.1958  , 1049.6564  ], dtype=float)
vmaxs = np.array([  790.,  1282.,  1864.,  3355.,  4615.,  6063.,  7778.,  9907.,
       11215., 12756., 14629., 17085., 19169., 21322., 24764., 27524.,
       32490.], dtype=float)
vmins = np.array([-6.5500e+02, -2.2000e+01,  6.3400e+02,  2.0480e+03,  3.1350e+03,
        4.3690e+03,  5.8310e+03,  7.6350e+03,  8.7370e+03,  1.0042e+04,
        1.1685e+04,  1.3939e+04,  1.5909e+04,  1.7737e+04,  2.0386e+04,
        2.2457e+04,  2.5908e+04], dtype=float)

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
            arr=None,
            start_idx: int = 0,
            end_idx: Optional[int] = None,
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

        if arr is None:
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
        self.mean = means[level] #11884.07 #self.data.mean()
        self.std = stds[level] #9119.372 #self.data.std()
        self.max = vmaxs[level]
        self.min = vmins[level]
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
        # NEW: window indexing / split support
        self.num_windows_total = 1 + self.max_start // self.stride

        if start_idx < 0 or start_idx >= self.num_windows_total:
            raise ValueError(f"start_idx {start_idx} out of range [0, {self.num_windows_total})")

        if end_idx is None:
            end_idx = self.num_windows_total
        if end_idx <= start_idx or end_idx > self.num_windows_total:
            raise ValueError(f"end_idx {end_idx} out of range ({start_idx}, {self.num_windows_total}]")

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_windows = self.end_idx - self.start_idx

        
    def __len__(self) -> int:
        # number of windows with given stride
        return self.num_windows #1 + self.max_start // self.stride

    def normalize(self, x):
        """
        x: torch.Tensor or np.ndarray
        """
        return (x - self.mean) / self.std

    def denormalize(self, x):
        """
        x: torch.Tensor or np.ndarray
        """
        return x * self.std + self.mean
         
    def __getitem__(self, idx: int):
        idx = idx + self.start_idx

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
    
    val_frac = 0.1 # get this from config tbd
    
    arr = np.load(data_files, mmap_mode="r")
    lat = np.load('utils/lat.npy') 
    lon = np.load('utils/lon.npy')
    vmin = vmins[int(levels)]
    vmax = vmaxs[int(levels)]
    # Compute number of windows total (no dataset needed)
    T_total = arr.shape[0]
    max_start = T_total - (input_len + target_len)
    if max_start < 0:
        raise ValueError(
            f"Not enough time steps ({T_total}) for input_len={input_len} + target_len={target_len}"
        )
    num_windows_total = 1 + (max_start // stride)

    split = int(num_windows_total * (1.0 - val_frac))


    print(f"get HGTSpatioTemporalDataset")
    train_ds = HGTSpatioTemporalDataset(
        data_files=data_files,
        arr=arr,
        input_len=input_len,
        target_len=target_len,
        stride=stride,
        level=int(levels),
        start_idx=0,
        end_idx=split,
    )
    
    val_ds = HGTSpatioTemporalDataset(
        data_files=data_files,
        arr=arr,                    # share memmap
        input_len=input_len,
        target_len=target_len,
        stride=stride,
        level=int(levels),
        start_idx=split,
        end_idx=num_windows_total,
    )
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    print(f"get the dataloader")
    train_loader = StatefulDataLoader(
        train_ds,
        batch_size=job_config.training.batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=job_config.training.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = StatefulDataLoader(
        val_ds,
        batch_size=job_config.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=job_config.training.num_workers,
        pin_memory=True,
        persistent_workers=(job_config.training.num_workers > 0),
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler, lat, lon, vmin, vmax #dataloader

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

    #loader = build_hgt_dataloader(job_config, rank, world_size)
    train_loader, val_loader, train_sampler, _ = build_hgt_dataloader(job_config, rank, world_size) 
    logger.info("Iterating over a few batches from HGT dataloader...")
    for i, batch in enumerate(train_loader):
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
