import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class RandomTokenDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        num_samples: int,
        seed: int = 0,
        dtype: torch.dtype = torch.long,
    ) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.seq_len,),
            generator=generator,
            dtype=self.dtype,
        )
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_random_token_dataloader(
    job_config,
    rank: int,
    world_size: int,
    seq_len: int,
    vocab_size: int,
    num_samples: int,
    seed: int = 0,
    dtype: torch.dtype = torch.long,
):
    dataset = RandomTokenDataset(
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
        seed=seed,
        dtype=dtype,
    )

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=job_config.training.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=job_config.training.num_workers,
        pin_memory=True,
        persistent_workers=0,
        drop_last=True,
    )

    return dataloader, sampler
