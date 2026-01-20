import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class RandomHGTDataset(Dataset):
    def __init__(
        self,
        input_len: int,
        target_len: int,
        channels: int,
        height: int,
        width: int,
        num_samples: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.input_len = input_len
        self.target_len = target_len
        self.channels = channels
        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.seed = seed
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        inputs = torch.randn(
            (self.input_len, self.channels, self.height, self.width),
            generator=generator,
            dtype=self.dtype,
        )
        targets = torch.randn(
            (self.target_len, self.channels, self.height, self.width),
            generator=generator,
            dtype=self.dtype,
        )
        return {"inputs": inputs, "targets": targets}


def build_random_hgt_dataloader(
    job_config,
    rank: int,
    world_size: int,
    input_len: int,
    target_len: int,
    channels: int,
    height: int,
    width: int,
    num_samples: int,
    val_num_samples: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    train_ds = RandomHGTDataset(
        input_len=input_len,
        target_len=target_len,
        channels=channels,
        height=height,
        width=width,
        num_samples=num_samples,
        seed=seed,
        dtype=dtype,
    )
    val_ds = RandomHGTDataset(
        input_len=input_len,
        target_len=target_len,
        channels=channels,
        height=height,
        width=width,
        num_samples=val_num_samples,
        seed=seed + 1,
        dtype=dtype,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = StatefulDataLoader(
        train_ds,
        batch_size=job_config.training.batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=job_config.training.num_workers,
        pin_memory=True,
        persistent_workers=(job_config.training.num_workers > 0),
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

    return train_loader, val_loader, train_sampler, val_sampler
