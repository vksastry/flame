dfile="/eagle/datascience/vsastry/projects/LatentTwinShared/hgt_all_new.npy"

import numpy as np

data = np.load(dfile, mmap_mode="r")  # (112500, 1, 73, 144)
mean = data.mean()
std  = data.std()
np.savez("hgt_stats.npz", mean=mean, std=std)
print("mean:", mean, "std:", std)
